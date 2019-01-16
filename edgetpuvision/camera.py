import threading

import numpy as np

from . import gstreamer
from .gst import *


def inference_pipeline(render_size, inference_size):
    size = max_inner_size(render_size, inference_size)
    return (
        Filter('glfilterbin', filter='glcolorscale'),
        Caps('video/x-raw', format='RGBA', width=size.width, height=size.height),
        Filter('videoconvert'),
        Caps('video/x-raw', format='RGB', width=size.width, height=size.height),
        Filter('videobox', autocrop=True),
        Caps('video/x-raw', width=inference_size.width, height=inference_size.height),
        Filter('appsink', name='appsink', emit_signals=True, max_buffers=1, drop=True, sync=False)
    )


def file_streaming_pipeline(filename, render_size, inference_size):
    return (
        Filter('filesrc', location=filename),
        Filter('qtdemux'),
        Filter('h264parse', config_interval=-1),
        Caps('video/x-h264', stream_format='byte-stream', profile='baseline', alignment='nal'),
        Tee(pins=((
          Queue(),
          Filter('vpudec'),
          inference_pipeline(render_size, inference_size),
        ), (
          Queue(),
          Filter('appsink', name='h264sink', emit_signals=True, max_buffers=1, drop=False, sync=False),
        )))
    )


def camera_streaming_pipeline(render_size, inference_size, profile, bitrate):
    size = max_inner_size(render_size, inference_size)
    return (
        Filter('v4l2src', device='/dev/video1'),
        Caps('video/x-raw', format='YUY2', width=640, height=360, framerate='15/1'),
        Tee(pins=((
          Queue(),
          inference_pipeline(render_size, inference_size)
        ), (
          Queue(),
          Filter('videoconvert'),
          Filter('x264enc',
                 speed_preset='ultrafast',
                 tune='zerolatency',
                 threads=4,
                 key_int_max=5,
                 bitrate=int(bitrate / 1000),  # kbit per second.
                 aud=False),
          Caps('video/x-h264', profile=profile),
          Filter('h264parse'),
          Caps('video/x-h264', stream_format='byte-stream', alignment='nal'),
          Filter('appsink', name='h264sink', emit_signals=True, max_buffers=1, drop=False, sync=False),
          # Tee(pins=((
          #     Queue(),
          #     Filter('appsink', name='h264sink', emit_signals=True, max_buffers=1, drop=False, sync=False)
          # ),(
          #     Queue(),
          #     Filter('vpudec'),
          #     Filter('kmssink', sync=False)
          # )))
        )))
    )


class InferenceCamera:
    def __init__(self, render_size, inference_size):
        self._render_size = Size(*render_size)
        self._inference_size = Size(*inference_size)
        self._loop = gstreamer.loop()
        self._thread = None
        self.on_image = None

    @property
    def resolution(self):
        return self._render_size

    def request_key_frame(self):
        pass

    def start_recording(self, obj, format, profile, inline_headers, bitrate, intra_period):
        size = min_outer_size(self._inference_size, self._render_size)
        view_box = center_inside(self._render_size, size)
        fps_counter = gstreamer.avg_fps_counter(30)

        def on_buffer(data, _):
            obj.write(data)

        def on_image(data, _):
            if self.on_image:
                self.on_image(np.frombuffer(data, dtype=np.uint8), next(fps_counter), size, view_box)

        signals = {
          'h264sink': {'new-sample': gstreamer.new_sample_callback(on_buffer)},
          'appsink': {'new-sample': gstreamer.new_sample_callback(on_image)},
        }

        pipeline = camera_streaming_pipeline(self._render_size, self._inference_size,
                                             profile=profile, bitrate=bitrate)

        self._thread = threading.Thread(target=gstreamer.run_pipeline,
                                        args=(self._loop, pipeline, signals))
        self._thread.start()

    def stop_recording(self):
        self._loop.quit()
        self._thread.join()
