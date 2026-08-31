"""GStreamer camera backend."""

import time

import cv2
import numpy as np


class GStreamerBackend:
    def __init__(self, config, preview_window_handle=None):
        self.config = config
        self.preview_window_handle = preview_window_handle
        self.pipeline = None
        self._appsink = None
        self._gst = None
        self._gst_app = None
        self._info = None

    def open(self, device):
        import gi
        gi.require_version("Gst", "1.0")
        gi.require_version("GstApp", "1.0")
        from gi.repository import Gst, GstApp

        Gst.init(None)
        self._gst = Gst
        self._gst_app = GstApp
        width = self.config.width
        height = self.config.height
        fps = self.config.requested_fps
        pipeline_desc = (
            f"v4l2src device={device} io-mode=2 ! "
            f"image/jpeg,width={width},height={height},framerate={fps}/1 ! "
            "nvv4l2decoder mjpeg=1 ! "
            "nvvidconv ! video/x-raw,format=BGRx ! "
            "appsink name=sink emit-signals=false max-buffers=1 drop=true sync=false "
            "qos=false enable-last-sample=false wait-on-eos=false"
        )
        self.pipeline = Gst.parse_launch(pipeline_desc)
        self._appsink = self.pipeline.get_by_name("sink")
        self.pipeline.set_state(Gst.State.PLAYING)
        self._info = type("CameraInfo", (), {
            "device": device,
            "backend": "GStreamer",
            "width": width,
            "height": height,
            "negotiated_fps": float(fps),
            "source_format": "MJPG",
            "output_format": "BGR",
        })()
        print("GStreamer 原生 appsink 已启动：Jetson 硬件 MJPEG 解码")
        return self._info

    @property
    def hardware_preview(self):
        return False

    def _read_native(self):
        # PyGObject on the Jetson image does not expose GstAppSink.try_pull_sample()
        # as a Python method; use the standard appsink signal API instead.
        sample = self._appsink.emit("try-pull-sample", self._gst.SECOND // 2)
        if sample is None:
            return False, None
        buffer = sample.get_buffer()
        ok, map_info = buffer.map(self._gst.MapFlags.READ)
        if not ok:
            return False, None
        try:
            caps = sample.get_caps().get_structure(0)
            width = caps.get_value("width")
            height = caps.get_value("height")
            frame = np.frombuffer(map_info.data, dtype=np.uint8).reshape((height, width, 4))
            return True, frame[:, :, :3].copy()
        finally:
            buffer.unmap(map_info)

    def read(self):
        return self._read_native()

    def release(self):
        if self.pipeline is not None:
            self.pipeline.set_state(self._gst.State.NULL)
            self.pipeline = None
        self._appsink = None
