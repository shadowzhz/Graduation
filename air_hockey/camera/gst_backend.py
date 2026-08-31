"""GStreamer 摄像头后端，优先原生 appsink，不行再退 OpenCV。"""

import cv2
import numpy as np

from .backend import CameraBackend
from .types import CameraInfo


class GStreamerBackend(CameraBackend):
    """Jetson 上 OpenCV 的 CAP_GSTREAMER 会多一层适配开销，
    所以优先用 gi.repository.Gst 直接拉 appsink。"""

    def __init__(self, config, preview_window_handle=None) -> None:
        super().__init__(config)
        self.pipeline = None
        self._appsink = None
        self._gst = None
        self._native = False
        self._native_pipeline = ""
        self._native_width = 0
        self._native_height = 0
        self._native_channels = 0
        self._preview_window_handle = preview_window_handle
        self._hardware_preview = False
        self._released = False

    @property
    def hardware_preview(self) -> bool:
        return self._hardware_preview

    @staticmethod
    def available() -> bool:
        if cv2.CAP_GSTREAMER > 0:
            return True
        try:
            import gi
            return True
        except ImportError:
            return False

    @staticmethod
    def _load_gst():
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        Gst.init(None)
        return Gst

    @staticmethod
    def _load_gst_video():
        import gi

        gi.require_version("GstVideo", "1.0")
        from gi.repository import GstVideo

        return GstVideo

    def open(self, device: str) -> CameraInfo:
        if not self.available():
            raise RuntimeError("GStreamer backend unavailable")
        try:
            return self._open_native(device)
        except Exception as native_error:
            self._release_native()
            if cv2.CAP_GSTREAMER <= 0:
                raise RuntimeError(f"原生 GStreamer 管道无法打开: {native_error}")
            print(f"原生 GStreamer appsink 不可用，回退到 OpenCV: {native_error}")
            return self._open_opencv(device)

    def _pipeline_descriptions(self, device):
        width = self.config.width
        height = self.config.height
        fps = int(self.config.requested_fps)
        source_format = self.config.pixel_format.upper()
        sink = (
            "appsink name=airhockeysink drop=true max-buffers=1 "
            "sync=false qos=false enable-last-sample=false wait-on-eos=false"
        )

        if source_format in {"MJPG", "JPEG"}:
            source = (
                f"v4l2src device={device} io-mode=2 ! "
                f"image/jpeg,width={width},height={height},framerate={fps}/1 ! "
                "nvv4l2decoder mjpeg=1 ! "
            )
            return (
                # Jetson 上 nvvidconv 输出 BGRx，读的时候再去掉 X 通道
                source + f"nvvidconv ! video/x-raw,format=BGRx ! {sink}",
                f"v4l2src device={device} io-mode=2 ! "
                f"image/jpeg,width={width},height={height},framerate={fps}/1 ! "
                f"jpegdec ! videoconvert ! "
                f"video/x-raw,format=BGR ! {sink}",
            )
        if source_format in {"YUYV", "YUY2"}:
            return (
                f"v4l2src device={device} io-mode=2 ! "
                f"video/x-raw,format=YUY2,width={width},height={height},"
                f"framerate={fps}/1 ! videoconvert ! video/x-raw,format=BGR ! {sink}",
            )
        raise ValueError(f"GStreamer 不支持输入格式: {self.config.pixel_format}")

    def _hardware_preview_description(self, device):
        """解码后 tee 一路给硬件显示，一路给 Python 检测。"""
        if self._preview_window_handle is None or self.config.pixel_format.upper() not in {"MJPG", "JPEG"}:
            return None
        width = self.config.width
        height = self.config.height
        fps = int(self.config.requested_fps)
        return (
            f"v4l2src device={device} io-mode=2 ! "
            f"image/jpeg,width={width},height={height},framerate={fps}/1 ! "
            "nvv4l2decoder mjpeg=1 ! tee name=split "
            "split. ! queue max-size-buffers=2 leaky=downstream ! "
            "nvvidconv ! nvegltransform ! "
            "nveglglessink name=preview_sink create-window=false sync=false "
            f"window-width={width} window-height={height} "
            "split. ! queue max-size-buffers=1 leaky=downstream ! "
            "nvvidconv ! video/x-raw,format=BGRx ! "
            "appsink name=airhockeysink drop=true max-buffers=1 sync=false "
            "qos=false enable-last-sample=false wait-on-eos=false"
        )

    def _open_native(self, device):
        Gst = self._load_gst()
        errors = []
        preview_description = self._hardware_preview_description(device)
        descriptions = ((preview_description, True),) if preview_description else ()
        descriptions += tuple((description, False) for description in self._pipeline_descriptions(device))
        for description, use_hardware_preview in descriptions:
            pipeline = None
            try:
                pipeline = Gst.parse_launch(description)
                appsink = pipeline.get_by_name("airhockeysink")
                if appsink is None:
                    raise RuntimeError("找不到 GStreamer appsink")
                if use_hardware_preview:
                    preview_sink = pipeline.get_by_name("preview_sink")
                    if preview_sink is None:
                        raise RuntimeError("找不到 GStreamer 硬件预览 sink")
                    GstVideo = self._load_gst_video()
                    GstVideo.VideoOverlay.set_window_handle(preview_sink, self._preview_window_handle)
                result = pipeline.set_state(Gst.State.PLAYING)
                if result == Gst.StateChangeReturn.FAILURE:
                    raise RuntimeError("GStreamer 管道无法进入 PLAYING 状态")
                result, state, _pending = pipeline.get_state(5 * Gst.SECOND)
                if result == Gst.StateChangeReturn.FAILURE or state != Gst.State.PLAYING:
                    raise RuntimeError(f"GStreamer 管道状态异常: {state.value_nick}")
                self.pipeline = pipeline
                self._appsink = appsink
                self._gst = Gst
                self._native = True
                self._hardware_preview = use_hardware_preview
                self._native_width = self.config.width
                self._native_height = self.config.height
                self._native_channels = 4 if "format=BGRx" in description else 3
                self._native_pipeline = (
                    "Jetson 硬件 MJPEG 解码"
                    if "nvv4l2decoder" in description
                    else "软件 JPEG 解码"
                )
                print(f"GStreamer 原生 appsink 已启动：{self._native_pipeline}")
                if self._hardware_preview:
                    print("Jetson 硬件预览已嵌入 GUI 视频区域")
                return self._set_info(device, self.config.width, self.config.height, self.config.requested_fps)
            except Exception as exc:
                errors.append(str(exc))
                if "nvv4l2decoder" in description:
                    print(f"GStreamer 硬件解码管道启动失败，尝试软件解码: {exc}")
                if pipeline is not None:
                    pipeline.set_state(Gst.State.NULL)
        raise RuntimeError("原生 GStreamer 管道无法打开: " + "; ".join(errors))

    def _open_opencv(self, device):
        descriptions = self._pipeline_descriptions(device)
        # OpenCV 的 appsink 不能带 name
        descriptions = tuple(description.replace(" name=airhockeysink", "") for description in descriptions)
        for description in descriptions:
            cap = cv2.VideoCapture(description, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                self.pipeline = cap
                self._native = False
                return self._set_info(
                    device,
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.config.width,
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.config.height,
                    cap.get(cv2.CAP_PROP_FPS) or self.config.requested_fps,
                )
            cap.release()
        raise RuntimeError("OpenCV GStreamer 管道无法打开")

    def _set_info(self, device, width, height, fps) -> CameraInfo:
        info = CameraInfo(
            device=device,
            backend="GStreamer",
            width=width,
            height=height,
            requested_fps=self.config.requested_fps,
            negotiated_fps=fps,
            source_format=self.config.pixel_format,
            output_format="BGR",
        )
        self.info = info
        return info

    def read(self):
        if self._released:
            return False, None
        if self._native:
            return self._read_native()
        ok, frame = self.pipeline.read()
        return (ok, self._as_bgr(frame)) if ok and frame is not None else (False, None)

    def _read_native(self):
        # 直接调用 GstAppSink 的 try_pull_sample，避免通用 signal emit 的额外分发开销。
        # timeout 保持有限，停止时不会长期卡死采集线程。
        sample = self._appsink.try_pull_sample(self._gst.SECOND // 2)
        if sample is None:
            return False, None
        buffer = sample.get_buffer()
        success, mapped = buffer.map(self._gst.MapFlags.READ)
        if not success:
            return False, None
        try:
            # 宽高和通道数在打开管道时缓存，避免每帧查 caps
            width = self._native_width
            height = self._native_height
            channels = self._native_channels
            expected_size = width * height * channels
            if len(mapped.data) < expected_size:
                raise RuntimeError("GStreamer BGR 缓冲区大小不足")
            # map 出来的内存在 unmap 前有效，必须复制成 NumPy 数组
            frame = np.frombuffer(mapped.data, dtype=np.uint8, count=expected_size)
            frame = frame.reshape((height, width, channels))
            if channels == 4:
                return True, frame[:, :, :3].copy()
            return True, frame.reshape((height, width, 3)).copy()
        finally:
            buffer.unmap(mapped)

    @staticmethod
    def _as_bgr(frame):
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] not in {3, 4}:
            raise RuntimeError("GStreamer backend must output BGR uint8 HxWx3")
        if frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame

    def _release_native(self) -> None:
        if self.pipeline is not None and self._native and self._gst is not None:
            self.pipeline.set_state(self._gst.State.NULL)
        self.pipeline = None
        self._appsink = None
        self._native = False
        self._native_pipeline = ""
        self._hardware_preview = False
        self._native_width = 0
        self._native_height = 0
        self._native_channels = 0

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        if self._native:
            self._release_native()
        elif self.pipeline is not None:
            self.pipeline.release()
            self.pipeline = None
