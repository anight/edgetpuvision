from .gst import *

def v4l2_camera(fmt):
    return (
        Filter('v4l2src', device=fmt.device),
        Caps('video/x-raw', format=fmt.pixel, width=fmt.size.width, height=fmt.size.height,
             framerate='%d/%d' % fmt.framerate),
    )

def display_sink(fullscreen, sync=False):
    return Filter('kmssink' if fullscreen else 'waylandsink', sync=sync),

def h264_sink(display_decoded=False):
    appsink = Filter('appsink', name='h264sink', emit_signals=True, max_buffers=1, drop=False, sync=False),

    if display_decoded:
        return Tee(pads=(
                   (Queue(), appsink),
                   (Queue(), Filter('vpudec'), Filter('kmssink', sync=False))
               ))

    return appsink

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

def image_display_pipeline(filename, render_size, inference_size, fullscreen):
    size = max_inner_size(render_size, inference_size)
    return (
        Filter('filesrc', location=filename),
        Filter('decodebin'),
        Filter('videoconvert'),
        Caps('video/x-raw', format='RGB'),
        Filter('imagefreeze'),
        Tee(pads=((
            Queue(max_size_buffers=1),
            Filter('videoconvert'),
            Filter('videoscale'),
            Filter('rsvgoverlay', name='overlay'),
            Caps('video/x-raw', width=render_size.width, height=render_size.height),
            display_sink(fullscreen),
        ),(
            Queue(max_size_buffers=1),
            Filter('videoconvert'),
            Filter('videoscale'),
            Caps('video/x-raw', width=size.width, height=size.height),
            Filter('videobox', autocrop=True),
            Caps('video/x-raw', width=inference_size.width, height=inference_size.height),
            Filter('appsink', name='appsink', emit_signals=True, max_buffers=1, drop=True, sync=False)
        )))
    )

def video_display_pipeline(filename, render_size, inference_size, fullscreen):
    return (
        Filter('filesrc', location=filename),
        Filter('qtdemux'),
        Filter('h264parse'),
        Filter('vpudec'),
        Filter('glupload'),
        Tee(pads=((
            Queue(max_size_buffers=1),
            Filter('glfilterbin', filter='glcolorscale'),
            Filter('rsvgoverlay', name='overlay'),
            Caps('video/x-raw', width=render_size.width, height=render_size.height),
            display_sink(fullscreen),
        ),(
            Queue(max_size_buffers=1, leaky='downstream'),
            inference_pipeline(render_size, inference_size),
        )))
    )

def camera_display_pipeline(render_size, inference_size, fullscreen):
    return (
        # TODO(dkovalev): Queue(max_size_buffers=1, leaky='downstream'),
        Filter('glupload'),
        Tee(pads=((
            Queue(max_size_buffers=1, leaky='downstream'),
            Filter('glfilterbin', filter='glcolorscale'),
            Filter('rsvgoverlay', name='overlay'),
            display_sink(fullscreen),
        ),(
            Queue(max_size_buffers=1, leaky='downstream'),
            inference_pipeline(render_size, inference_size),
        )))
    )

def video_streaming_pipeline(filename, render_size, inference_size):
    return (
        Filter('filesrc', location=filename),
        Filter('qtdemux'),
        Tee(pads=((
          Queue(max_size_buffers=1),
          Filter('h264parse'),
          Filter('vpudec'),
          inference_pipeline(render_size, inference_size),
        ), (
          Queue(max_size_buffers=1),
          Filter('h264parse'),
          Caps('video/x-h264', stream_format='byte-stream', alignment='nal'),
          h264_sink()
        )))
    )

def camera_streaming_pipeline(profile, bitrate, render_size, inference_size):
    size = max_inner_size(render_size, inference_size)
    return (
        Tee(pads=((
          Queue(),
          inference_pipeline(render_size, inference_size)
        ), (
          Queue(max_size_buffers=1, leaky='downstream'),
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
          h264_sink()
        )))
    )