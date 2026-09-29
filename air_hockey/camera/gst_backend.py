"""GStreamer camera backend with explicitly owned BGR frames."""

import time
from fractions import Fraction

import cv2
import numpy as np

from .types import CameraInfo


class GStreamerBackend:
    def __init__(self, config):
        self.config = config
        self.pipeline = None
        self._appsink = None
        self._bus = None
        self._gst = None
        self._device = None
        self.info = None
        self.last_read_ms = 0.0
        self.last_convert_ms = 0.0
        self.last_pts_ns = None

    def open(self, device):
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        Gst.init(None)
        self._gst = Gst
        self._device = device
        self.info = None
        self.last_read_ms = self.last_convert_ms = 0.0
        self.last_pts_ns = None
        width = self.config.width
        height = self.config.height
        fps = Fraction(str(self.config.requested_fps)).limit_denominator(1000)
        pipeline_desc = (
            f"v4l2src device={device} io-mode=2 ! "
            f"image/jpeg,width={width},height={height},framerate={fps.numerator}/{fps.denominator} ! "
            "nvv4l2decoder mjpeg=1 ! "
            "nvvidconv ! video/x-raw,format=BGRx ! "
            "appsink name=sink emit-signals=false max-buffers=1 drop=true sync=false "
            "qos=false enable-last-sample=false wait-on-eos=false"
        )
        try:
            self.pipeline = Gst.parse_launch(pipeline_desc)
            self._appsink = self.pipeline.get_by_name("sink")
            if self._appsink is None:
                raise RuntimeError("GStreamer pipeline 中没有 appsink")
            self._bus = self.pipeline.get_bus()
            result = self.pipeline.set_state(Gst.State.PLAYING)
            if result == Gst.StateChangeReturn.FAILURE:
                self._check_bus()
                raise RuntimeError("GStreamer set_state(PLAYING) 失败")
            self._check_bus()
        except Exception:
            self.release()
            raise

    def _check_bus(self):
        if self._bus is None:
            return
        gst = self._gst
        while True:
            message = self._bus.pop_filtered(gst.MessageType.ERROR | gst.MessageType.EOS)
            if message is None:
                return
            if message.type == gst.MessageType.ERROR:
                error, debug = message.parse_error()
                source = message.src.get_name() if message.src is not None else "pipeline"
                raise RuntimeError(
                    f"GStreamer {self._device} ERROR ({source}): {error}"
                    + (f"; {debug}" if debug else "")
                )
            raise RuntimeError(f"GStreamer {self._device} EOS：摄像头流已结束")

    def _record_info(self, sample, width, height):
        if self.info is not None:
            return
        structure = sample.get_caps().get_structure(0)
        fps = structure.get_value("framerate")
        if hasattr(fps, "num") and hasattr(fps, "denom"):
            numerator, denominator = fps.num, fps.denom
        else:
            # GI installations may expose Gst.Fraction as a pair.
            try:
                numerator, denominator = fps
            except (TypeError, ValueError):
                raise RuntimeError(f"GStreamer caps 中帧率无效: {fps!r}") from None
        if not numerator or not denominator:
            raise RuntimeError(f"GStreamer caps 中帧率无效: {fps!r}")
        if structure.get_value("format") != "BGRx":
            raise RuntimeError("GStreamer appsink 未协商 BGRx 输出")
        self.info = CameraInfo(
            device=self._device,
            backend="GStreamer",
            width=width,
            height=height,
            requested_fps=self.config.requested_fps,
            negotiated_fps=float(numerator) / denominator,
            source_format="MJPG",
            output_format="BGR",
        )

    def read(self):
        if self._appsink is None:
            raise RuntimeError("GStreamer appsink 未启动")
        started = time.perf_counter()
        self._check_bus()
        # The appsink signal works on Jetson images where GstAppSink's Python
        # try_pull_sample method is not exposed; no GstApp GI namespace needed.
        sample = self._appsink.emit("try-pull-sample", self._gst.SECOND // 2)
        if sample is None:
            self._check_bus()
            return False, None
        buffer = sample.get_buffer()
        if buffer is None:
            raise RuntimeError(f"GStreamer {self._device} 返回空 buffer")
        caps = sample.get_caps()
        if caps is None or caps.get_size() == 0:
            raise RuntimeError(f"GStreamer {self._device} sample 缺少协商 caps")
        structure = caps.get_structure(0)
        width, height = structure.get_value("width"), structure.get_value("height")
        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            raise RuntimeError(f"GStreamer {self._device} caps 尺寸无效: {width}x{height}")
        self._record_info(sample, width, height)
        ok, mapped = buffer.map(self._gst.MapFlags.READ)
        if not ok:
            raise RuntimeError(f"GStreamer {self._device} buffer map 失败")
        try:
            # GstVideoMeta may specify padded strides/offsets. Without querying
            # them, only tightly packed BGRx is safe to interpret as pixels.
            expected = height * width * 4
            if len(mapped.data) != expected:
                raise RuntimeError(
                    f"GStreamer {self._device} BGRx buffer 大小 {len(mapped.data)} != {expected}; "
                    "可能存在 stride/offset，不能安全转换"
                )
            pixels = np.frombuffer(mapped.data, dtype=np.uint8).reshape((height, width, 4))
            convert_started = time.perf_counter()
            image = cv2.cvtColor(pixels, cv2.COLOR_BGRA2BGR)
            convert_ms = (time.perf_counter() - convert_started) * 1000.0
        finally:
            buffer.unmap(mapped)
        self.last_read_ms = (time.perf_counter() - started) * 1000.0
        self.last_convert_ms = convert_ms
        self.last_pts_ns = None if buffer.pts == self._gst.CLOCK_TIME_NONE else int(buffer.pts)
        return True, image

    def release(self):
        if self.pipeline is not None:
            try:
                self.pipeline.set_state(self._gst.State.NULL)
            finally:
                self.pipeline = None
                self._appsink = None
                self._bus = None
