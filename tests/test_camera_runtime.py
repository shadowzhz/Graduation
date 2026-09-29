"""Camera startup and runtime failures without a physical GStreamer device."""

import threading
import time
from types import SimpleNamespace
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
    structure = SimpleNamespace(get_value=lambda key: {
        "width": 2, "height": 1, "framerate": SimpleNamespace(num=30, denom=1),
        "format": "BGRx",
    }[key])
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


def test_gst_rejects_unknown_padded_stride():
    from air_hockey.camera.gst_backend import GStreamerBackend

    structure = SimpleNamespace(get_value=lambda key: {
        "width": 2, "height": 1, "framerate": SimpleNamespace(num=30, denom=1),
        "format": "BGRx",
    }[key])
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
