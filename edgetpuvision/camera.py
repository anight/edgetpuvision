import os
import threading

import numpy as np

from . import gstreamer
from . import pipelines

from .gst import *

class Camera:
    def __init__(self, render_size, inference_size):
        self._layout = gstreamer.make_layout(Size(*inference_size), Size(*render_size))

        self._loop = gstreamer.loop()
        self._thread = None

        self.on_image = None

    @property
    def resolution(self):
        return self._layout.render_size

    def request_key_frame(self):
        pass

    def start_recording(self, obj, format, profile, inline_headers, bitrate, intra_period):
        def on_buffer(data, _):
            obj.write(data)

        def on_image(data, _):
            if self.on_image:
                self.on_image(np.frombuffer(data, dtype=np.uint8), self._layout)

        signals = {
          'h264sink': {'new-sample': gstreamer.new_sample_callback(on_buffer)},
          'appsink': {'new-sample': gstreamer.new_sample_callback(on_image)},
        }

        pipeline = self.make_pipeline(format, profile, inline_headers, bitrate, intra_period)

        self._thread = threading.Thread(target=gstreamer.run_pipeline,
                                        args=(self._loop, pipeline, signals))
        self._thread.start()

    def stop_recording(self):
        self._loop.quit()
        self._thread.join()

    def make_pipeline(self, fmt, profile, inline_headers, bitrate, intra_period):
        raise NotImplemented

class FileCamera(Camera):
    def __init__(self, filename, inference_size):
        info = gstreamer.get_video_info(filename)
        super().__init__((info.get_width(), info.get_height()), inference_size)
        self._filename = filename

    def make_pipeline(self, fmt, profile, inline_headers, bitrate, intra_period):
        return pipelines.video_streaming_pipeline(self._filename, self._layout)

class V4L2Camera(Camera):
    def __init__(self, fmt, inference_size):
        super().__init__(fmt.size, inference_size)
        self._fmt = fmt

    def make_pipeline(self, fmt, profile, inline_headers, bitrate, intra_period):
        return pipelines.camera_streaming_pipeline(self._fmt, profile, bitrate, self._layout)

def make_camera(source, inference_size):
    fmt = parse_format(source)
    if fmt:
        return V4L2Camera(fmt, inference_size)

    filename = os.path.expanduser(source)
    if os.path.isfile(filename):
        return FileCamera(filename, inference_size)

    return None
