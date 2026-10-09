"""Camera startup and runtime failures without a physical GStreamer device."""

import threading
import time
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np

from air_hockey.camera import CameraConfig, CameraManager
from air_hockey.camera import manager as camera_manager


def test_start_tries_next_device_after_first_frame_failure():
    attempts = []
    gate = threading.Event()
    image = object()

    class Backend:
        def __init__(self, config):
            self.config = config
            self.info = None
            self.last_read_ms = 12.5
            self.last_convert_ms = 1.25
            self.last_pts_ns = 123456

        def open(self, device):
            self.device = device
            attempts.append(device)
            self.calls = 0

        def read(self):
            if self.device == "/dev/video2":
                raise RuntimeError("decoder ERROR")
            self.calls += 1
            if self.calls > 1:
                time.sleep(0.02)
                return False, None
            self.info = SimpleNamespace(
                device=self.device, width=640, height=480, negotiated_fps=30.0
            )
            gate.wait(1.0)  # keep the capture thread alive until startup observes first frame
            return True, image

        def release(self):
            pass

    backend_patch = patch.object(camera_manager, "GStreamerBackend", Backend)
    devices_patch = patch.object(camera_manager.glob, "glob", lambda _: ["/dev/video10", "/dev/video2"])
    backend_patch.start()
    devices_patch.start()
    manager = CameraManager(CameraConfig())
    try:
        # Gate is set by a short timer so start cannot observe the second fake
        # device's frame before its first fake device has failed.
        timer = threading.Timer(0.02, gate.set)
        timer.start()
        manager.start(timeout=0.5)
        timer.join(1.0)
        assert attempts == ["/dev/video2", "/dev/video10"]
        assert manager.info.device == "/dev/video10"
        assert manager.info.negotiated_fps == 30.0
        frame = manager.get_latest_frame()
        assert frame.image is image
        assert (frame.capture_read_ms, frame.color_convert_ms, frame.gst_pts_ns) == (
            12.5, 1.25, 123456
        )
        assert manager.error is None
    finally:
        manager.stop()
        devices_patch.stop()
        backend_patch.stop()


def test_runtime_read_failure_stops_capture_and_reports_error():
    next_read = threading.Event()

    class Backend:
        def __init__(self, config):
            self.info = SimpleNamespace(device="/dev/video0")
            self.calls = 0
            self.last_read_ms = 0.0
            self.last_convert_ms = 0.0
            self.last_pts_ns = None

        def open(self, device):
            pass

        def read(self):
            self.calls += 1
            if self.calls == 1:
                return True, object()
            next_read.wait(1.0)
            raise RuntimeError("GStreamer ERROR: camera disconnected")

        def release(self):
            pass

    backend_patch = patch.object(camera_manager, "GStreamerBackend", Backend)
    backend_patch.start()
    manager = CameraManager(CameraConfig(device="/dev/video0"))
    try:
        manager.start(timeout=0.5)
        assert manager.error is None
        next_read.set()
        manager._thread.join(1.0)
        assert not manager.is_running()
        assert isinstance(manager.error, RuntimeError)
        assert "camera disconnected" in str(manager.error)
        assert manager.get_latest_frame() is not None
    finally:
        next_read.set()
        manager.stop()
        backend_patch.stop()


def test_gst_selects_software_decode_without_jetson_plugins():
    from air_hockey.camera.gst_backend import GStreamerBackend

    for jetson_plugins in (False, True):
        descriptions = []

        class Pipeline:
            def get_by_name(self, name):
                return object()

            def get_bus(self):
                return SimpleNamespace(pop_filtered=lambda mask: None)

            def set_state(self, state):
                return 1

        pipeline = Pipeline()
        gi = ModuleType("gi")
        gi.__path__ = []
        gi.require_version = lambda *args: None
        repository = ModuleType("gi.repository")
        gst = SimpleNamespace(
            init=lambda *args: None,
            ElementFactory=SimpleNamespace(find=lambda name: object() if jetson_plugins else None),
            parse_launch=lambda description: (descriptions.append(description) or pipeline),
            State=SimpleNamespace(PLAYING=1, NULL=0),
            StateChangeReturn=SimpleNamespace(FAILURE=-1),
            MessageType=SimpleNamespace(ERROR=1, EOS=2),
        )
        repository.Gst = gst
        with patch.dict("sys.modules", {"gi": gi, "gi.repository": repository}):
            backend = GStreamerBackend(CameraConfig())
            backend.open("/dev/video2")
            backend.release()

        description = descriptions[0]
        if jetson_plugins:
            assert "nvv4l2decoder mjpeg=1 ! nvvidconv" in description
            assert "jpegdec" not in description
        else:
            assert "jpegdec ! videoconvert" in description
            assert "nvv4l2decoder" not in description and "nvvidconv" not in description


def test_gst_bus_eos_and_error_are_not_silent():
    from air_hockey.camera.gst_backend import GStreamerBackend

    gst = SimpleNamespace(MessageType=SimpleNamespace(ERROR=1, EOS=2))

    class Bus:
        def __init__(self, message):
            self.message = message

        def pop_filtered(self, mask):
            message, self.message = self.message, None
            return message

    backend = GStreamerBackend(CameraConfig())
    backend._gst = gst
    backend._device = "/dev/video0"
    backend._bus = Bus(SimpleNamespace(type=2))
    try:
        backend._check_bus()
    except RuntimeError as exc:
        assert "EOS" in str(exc)
    else:
        raise AssertionError("GStreamer EOS must stop capture")
    backend._bus = Bus(SimpleNamespace(
        type=1, src=None, parse_error=lambda: ("device lost", "source stopped")
    ))
    try:
        backend._check_bus()
    except RuntimeError as exc:
        assert "device lost" in str(exc) and "source stopped" in str(exc)
    else:
        raise AssertionError("GStreamer ERROR must stop capture")


def test_gst_sample_caps_diagnostics_and_pixel_ownership():
    from air_hockey.camera.gst_backend import GStreamerBackend

    raw = bytearray([3, 7, 11, 0, 13, 17, 19, 0])
    def get_value(key):
        if key == "framerate":
            raise TypeError("unknown type GstFraction")
        return {"width": 2, "height": 1, "format": "BGRx"}[key]

    structure = SimpleNamespace(
        get_value=get_value,
        get_fraction=lambda key: (True, 30, 1) if key == "framerate" else (False, 0, 0),
    )
    caps = SimpleNamespace(get_size=lambda: 1, get_structure=lambda index: structure)

    class Buffer:
        pts = 123456789
        unmapped = False

        def map(self, flags):
            return True, SimpleNamespace(data=raw)

        def unmap(self, mapped):
            self.unmapped = True

    buffer = Buffer()
    sample = SimpleNamespace(get_caps=lambda: caps, get_buffer=lambda: buffer)
    backend = GStreamerBackend(CameraConfig(requested_fps=200.0))
    backend._gst = SimpleNamespace(SECOND=1_000_000_000, CLOCK_TIME_NONE=2**64 - 1,
                                   MapFlags=SimpleNamespace(READ=1))
    backend._device = "/dev/video0"
    backend._appsink = SimpleNamespace(emit=lambda signal, timeout: sample)
    ok, image = backend.read()

    assert ok and buffer.unmapped
    assert (backend.info.width, backend.info.height, backend.info.negotiated_fps) == (2, 1, 30.0)
    assert backend.info.requested_fps == 200.0
    assert backend.last_pts_ns == 123456789
    assert backend.last_read_ms >= backend.last_convert_ms >= 0.0
    np.testing.assert_array_equal(image, np.array([[[3, 7, 11], [13, 17, 19]]], dtype=np.uint8))
    raw[:] = bytes(len(raw))
    assert image[0, 0, 0] == 3  # frame owns pixels beyond GstBuffer unmap


def test_gst_rejects_invalid_negotiated_framerate():
    from air_hockey.camera.gst_backend import GStreamerBackend

    for fraction in ((False, 0, 0), (True, 30, 0), (True, -1, 1)):
        structure = SimpleNamespace(
            get_fraction=lambda key: fraction,
            get_value=lambda key: "BGRx",
        )
        sample = SimpleNamespace(get_caps=lambda: SimpleNamespace(get_structure=lambda index: structure))
        backend = GStreamerBackend(CameraConfig())
        try:
            backend._record_info(sample, 1280, 720)
        except RuntimeError as exc:
            assert "帧率无效" in str(exc)
        else:
            raise AssertionError("invalid negotiated FPS must not be reported as camera info")


def test_gst_rejects_unknown_padded_stride():
    from air_hockey.camera.gst_backend import GStreamerBackend

    structure = SimpleNamespace(
        get_value=lambda key: {"width": 2, "height": 1, "format": "BGRx"}[key],
        get_fraction=lambda key: (True, 30, 1),
    )
    caps = SimpleNamespace(get_size=lambda: 1, get_structure=lambda index: structure)
    buffer = SimpleNamespace(
        pts=0, map=lambda flags: (True, SimpleNamespace(data=b"\0" * 12)),
        unmap=lambda mapped: None,
    )
    sample = SimpleNamespace(get_caps=lambda: caps, get_buffer=lambda: buffer)
    backend = GStreamerBackend(CameraConfig())
    backend._gst = SimpleNamespace(SECOND=1_000_000_000, MapFlags=SimpleNamespace(READ=1))
    backend._device = "/dev/video0"
    backend._appsink = SimpleNamespace(emit=lambda signal, timeout: sample)
    try:
        backend.read()
    except RuntimeError as exc:
        assert "stride/offset" in str(exc)
    else:
        raise AssertionError("padded frames must not be interpreted as tightly packed")


def test_opencv_backend_uses_actual_directshow_frame_mode():
    from air_hockey.camera import opencv_backend

    image = np.zeros((480, 640, 3), dtype=np.uint8)
    fourcc = opencv_backend.cv2.VideoWriter_fourcc(*"MJPG")

    class Capture:
        def __init__(self):
            self.settings = {}
            self.released = False

        def isOpened(self):
            return True

        def set(self, prop, value):
            self.settings[prop] = value
            return True

        def read(self):
            return True, image

        def get(self, prop):
            return {opencv_backend.cv2.CAP_PROP_FPS: 30.0,
                    opencv_backend.cv2.CAP_PROP_FOURCC: fourcc}[prop]

        def release(self):
            self.released = True

    capture = Capture()
    calls = []
    backend = opencv_backend.OpenCVBackend(CameraConfig())
    with patch.object(opencv_backend.cv2, "VideoCapture",
                      lambda index, api: (calls.append((index, api)) or capture)):
        backend.open("2")
        ok, frame = backend.read()
        backend.release()

    assert calls == [(2, opencv_backend.cv2.CAP_DSHOW)]
    assert capture.settings == {
        opencv_backend.cv2.CAP_PROP_FOURCC: fourcc,
        opencv_backend.cv2.CAP_PROP_FRAME_WIDTH: 1280,
        opencv_backend.cv2.CAP_PROP_FRAME_HEIGHT: 720,
        opencv_backend.cv2.CAP_PROP_FPS: 200.0,
    }
    assert ok and frame is image
    assert (backend.info.width, backend.info.height, backend.info.negotiated_fps) == (640, 480, 30.0)
    assert backend.info.backend == "OpenCV/DirectShow"
    assert (backend.info.source_format, backend.info.output_format) == ("MJPG", "BGR")
    assert capture.released


def test_windows_camera_manager_starts_default_directshow_device():
    from air_hockey.camera import opencv_backend

    image = np.zeros((360, 640, 3), dtype=np.uint8)

    class Capture:
        def __init__(self):
            self.reads = 0
            self.released = False

        def isOpened(self):
            return True

        def set(self, prop, value):
            return True

        def read(self):
            self.reads += 1
            if self.reads == 1:
                return True, image
            time.sleep(0.001)
            return False, None

        def get(self, prop):
            if prop == opencv_backend.cv2.CAP_PROP_FPS:
                return 30.0
            if prop == opencv_backend.cv2.CAP_PROP_FOURCC:
                return 0.0
            raise AssertionError(f"unexpected camera property {prop}")

        def release(self):
            self.released = True

    capture = Capture()
    calls = []
    manager = CameraManager(CameraConfig())
    with patch.object(camera_manager.sys, "platform", "win32"), patch.object(
        opencv_backend.cv2, "VideoCapture",
        lambda index, api: (calls.append((index, api)) or capture),
    ):
        try:
            manager.start(timeout=0.5)
            frame = manager.get_latest_frame()
            assert calls == [(0, opencv_backend.cv2.CAP_DSHOW)]
            assert manager.info.device == "0"
            assert manager.info.backend == "OpenCV/DirectShow"
            assert (frame.image.shape[1], frame.image.shape[0]) == (640, 360)
            assert frame.gst_pts_ns is None
        finally:
            manager.stop()
    assert capture.released


def test_opencv_backend_rejects_linux_paths_and_releases_failed_open():
    from air_hockey.camera import opencv_backend

    class Capture:
        released = False

        def isOpened(self):
            return False

        def release(self):
            self.released = True

    capture = Capture()
    calls = []
    backend = opencv_backend.OpenCVBackend(CameraConfig())
    with patch.object(opencv_backend.cv2, "VideoCapture",
                      lambda index, api: (calls.append((index, api)) or capture)):
        try:
            backend.open("/dev/video2")
        except ValueError as exc:
            assert "编号" in str(exc)
        else:
            raise AssertionError("DirectShow must reject a Linux device path")
        assert calls == []
        try:
            backend.open("0")
        except RuntimeError as exc:
            assert "无法打开摄像头 0" in str(exc)
        else:
            raise AssertionError("an unopened camera must fail startup")
    assert calls == [(0, opencv_backend.cv2.CAP_DSHOW)]
    assert capture.released


def test_opencv_backend_rejects_non_bgr_frames():
    from air_hockey.camera import opencv_backend

    class Capture:
        def isOpened(self):
            return True

        def set(self, prop, value):
            return True

        def read(self):
            return True, np.zeros((2, 2), dtype=np.uint8)

        def release(self):
            pass

    backend = opencv_backend.OpenCVBackend(CameraConfig())
    with patch.object(opencv_backend.cv2, "VideoCapture", lambda index, api: Capture()):
        backend.open("0")
        try:
            backend.read()
        except RuntimeError as exc:
            assert "BGR uint8" in str(exc)
        else:
            raise AssertionError("grayscale frames must not enter the BGR vision pipeline")
        finally:
            backend.release()
