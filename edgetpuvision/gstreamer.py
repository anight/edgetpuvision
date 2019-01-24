import collections
import contextlib
import fcntl
import functools
import os
import queue
import sys
import termios
import threading
import time

import numpy as np

import gi
gi.require_version('GLib', '2.0')
gi.require_version('GObject', '2.0')
gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')
gi.require_version('GstPbutils', '1.0')

from gi.repository import GLib, GObject, Gst, GstBase

GObject.threads_init()
Gst.init(None)

from gi.repository import GstPbutils  # Must be called after Gst.init().

from PIL import Image

from .gst import *

COMMAND_SAVE_FRAME = ' '
COMMAND_PRINT_INFO = 'p'

def set_nonblocking(fd):
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    return fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

@contextlib.contextmanager
def term_raw_mode(fd):
    old = termios.tcgetattr(fd)
    new = termios.tcgetattr(fd)
    new[3] = new[3] & ~(termios.ICANON | termios.ECHO)
    termios.tcsetattr(fd, termios.TCSANOW, new)
    try:
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSAFLUSH, old)

def get_nowait(q):
    try:
        return q.get_nowait()
    except queue.Empty:
        return None

@contextlib.contextmanager
def Worker(process, maxsize=0):
    commands = queue.Queue(maxsize)

    def run():
        while True:
            args = commands.get()
            if args is None:
                break
            process(*args)
            commands.task_done()

    thread = threading.Thread(target=run)
    thread.start()
    try:
        yield commands
    finally:
        commands.put(None)
        thread.join()

def sink(fullscreen, sync=False):
    return Filter('kmssink' if fullscreen else 'waylandsink', sync=sync),

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

def image_file_pipeline(filename, render_size, inference_size, fullscreen):
    size = max_inner_size(render_size, inference_size)
    return (
        Filter('filesrc', location=filename),
        Filter('decodebin'),
        Filter('imagefreeze'),
        Tee(pads=((
            Queue(max_size_buffers=1),
            Filter('videoconvert'),
            Filter('videoscale'),
            Filter('rsvgoverlay', name='overlay'),
            Caps('video/x-raw', width=render_size.width, height=render_size.height),
            sink(fullscreen),
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

def video_file_pipeline(filename, render_size, inference_size, fullscreen):
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
            sink(fullscreen),
        ),(
            Queue(max_size_buffers=1, leaky='downstream'),
            inference_pipeline(render_size, inference_size),
        )))
    )

# v4l2-ctl --list-formats-ext --device /dev/video1
def v4l2_camera(fmt):
    return (
        Filter('v4l2src', device=fmt.device),
        Caps('video/x-raw', format=fmt.pixel, width=fmt.size.width, height=fmt.size.height,
             framerate='%d/%d' % fmt.framerate),
    )

def video_camera_pipeline(render_size, inference_size, fullscreen):
    return (
        # TODO(dkovalev): Queue(max_size_buffers=1, leaky='downstream'),
        Filter('glupload'),
        Tee(pads=((
            Queue(max_size_buffers=1, leaky='downstream'),
            Filter('glfilterbin', filter='glcolorscale'),
            Filter('rsvgoverlay', name='overlay'),
            sink(fullscreen),
        ),(
            Queue(max_size_buffers=1, leaky='downstream'),
            inference_pipeline(render_size, inference_size),
        )))
    )

def h264sink(display_decoded=False):
    appsink = Filter('appsink', name='h264sink', emit_signals=True, max_buffers=1, drop=False, sync=False),

    if display_decoded:
        return Tee(pads=(
                   (Queue(), appsink),
                   (Queue(), Filter('vpudec'), Filter('kmssink', sync=False))
               ))

    return appsink

def file_streaming_pipeline(filename, render_size, inference_size):
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
          h264sink()
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
          h264sink()
        )))
    )

def save_frame(rgb, size, overlay=None, ext='png'):
    tag = '%010d' % int(time.monotonic() * 1000)
    img = Image.frombytes('RGB', size, rgb, 'raw')
    name = 'img-%s.%s' % (tag, ext)
    img.save(name)
    print('Frame saved as "%s"' % name)
    if overlay:
        name = 'img-%s.svg' % tag
        with open(name, 'w') as f:
            f.write(overlay)
        print('Overlay saved as "%s"' % name)

def avg_fps_counter(window_size):
    window = collections.deque(maxlen=window_size)
    prev = time.monotonic()
    yield 0.0  # First fps value.

    while True:
        curr = time.monotonic()
        window.append(curr - prev)
        prev = curr
        yield len(window) / sum(window)

def caps_size(caps):
    structure = caps.get_structure(0)
    return Size(structure.get_value('width'),
                structure.get_value('height'))

def get_video_info(filename):
    #Command line: gst-discoverer-1.0 -v ~/cars_highway.mp4
    uri = 'file://' + filename
    discoverer = GstPbutils.Discoverer()
    info = discoverer.discover_uri(uri)

    streams = info.get_video_streams()
    assert len(streams) == 1
    return streams[0]

def loop():
    return GLib.MainLoop.new(None, False)

@contextlib.contextmanager
def pull_sample(sink):
    sample = sink.emit('pull-sample')
    buf = sample.get_buffer()

    result, mapinfo = buf.map(Gst.MapFlags.READ)
    if result:
        yield sample, mapinfo.data
    buf.unmap(mapinfo)

def new_sample_callback(process):
    def callback(sink, pipeline):
        with pull_sample(sink) as (sample, data):
            process(data, caps_size(sample.get_caps()))
        return Gst.FlowReturn.OK
    return callback

def on_bus_message(bus, message, loop):
    if message.type == Gst.MessageType.EOS:
        loop.quit()
    elif message.type == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        sys.stderr.write('Warning: %s: %s\n' % (err, debug))
    elif message.type == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        sys.stderr.write('Error: %s: %s\n' % (err, debug))
        loop.quit()
    return True

def run_pipeline(loop, pipeline, signals):
    # Create pipeline
    pipeline = describe(pipeline)
    print(pipeline)
    pipeline = Gst.parse_launch(pipeline)

    # Attach signals
    for name, signals in signals.items():
        component = pipeline.get_by_name(name)
        if component:
            for signal_name, signal_handler in signals.items():
                component.connect(signal_name, signal_handler, pipeline)

    # Set up a pipeline bus watch to catch errors.
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect('message', on_bus_message, loop)

    # Run pipeline.
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.set_state(Gst.State.NULL)


def on_keypress(fd, flags, commands):
    for ch in sys.stdin.read():
        commands.put(ch)
    return True

def on_new_sample(sink, pipeline, render_overlay, render_size, images, commands, fps_counter):
    with pull_sample(sink) as (sample, data):
        inference_rate = next(fps_counter)
        custom_command = None
        save_frame = False

        command = get_nowait(commands)
        if command == COMMAND_SAVE_FRAME:
            save_frame = True
        elif command == COMMAND_PRINT_INFO:
            print('Timestamp: %.2f' % time.monotonic())
            print('Inference FPS: %s' % inference_rate)
            print('Render size: %d x %d' % render_size)
            print('Inference size: %d x %d' % caps_size(sample.get_caps()))
        else:
            custom_command = command

        svg = render_overlay(np.frombuffer(data, dtype=np.uint8),
                             inference_rate=inference_rate,
                             command=custom_command)
        overlay = pipeline.get_by_name('overlay')
        overlay.set_property('data', svg)

        if save_frame:
            images.put((data, caps_size(sample.get_caps()), svg))

    return Gst.FlowReturn.OK


def run_gen(render_overlay_gen, *, source, downscale, fullscreen):
    inference_size = render_overlay_gen.send(None)  # Initialize.
    return run(inference_size,
        lambda tensor, size, window, inference_rate, command:
            render_overlay_gen.send((tensor, size, window, inference_rate, command)),
        source=source,
        downscale=downscale,
        fullscreen=fullscreen)

def run(inference_size, render_overlay, *, source, downscale, fullscreen):
    fmt = parse_format(source)
    if fmt:
        run_camera(inference_size, render_overlay, fmt, fullscreen)
        return True

    filename = os.path.expanduser(source)
    if os.path.isfile(filename):
        run_file(inference_size, render_overlay,
                 filename=filename,
                 downscale=downscale,
                 fullscreen=fullscreen)
        return True

    return False

def run_camera(inference_size, render_overlay, fmt, fullscreen):
    inference_size = Size(*inference_size)

    camera = v4l2_camera(fmt)
    caps = next(x for x in camera if isinstance(x, Caps))
    render_size = Size(caps.width, caps.height)
    pipeline = camera + video_camera_pipeline(render_size, inference_size, fullscreen)
    return run_loop(pipeline, inference_size, render_size, render_overlay)


def run_file(inference_size, render_overlay, *, filename, downscale, fullscreen):
    inference_size = Size(*inference_size)

    info = get_video_info(filename)
    render_size = Size(info.get_width(), info.get_height()) / downscale
    if info.is_image():
        pipeline = image_file_pipeline(filename, render_size, inference_size, fullscreen)
    else:
        pipeline = video_file_pipeline(filename, render_size, inference_size, fullscreen)

    return run_loop(pipeline, inference_size, render_size, render_overlay)


def run_loop(pipeline, inference_size, render_size, render_overlay):
    loop = GLib.MainLoop()
    commands = queue.Queue()

    with contextlib.ExitStack() as stack:
        images = stack.enter_context(Worker(save_frame))

        if sys.stdin.isatty():
            set_nonblocking(sys.stdin.fileno())
            GLib.io_add_watch(sys.stdin.fileno(), GLib.IO_IN, on_keypress, commands)
            stack.enter_context(term_raw_mode(sys.stdin.fileno()))

        size = min_outer_size(inference_size, render_size)
        window = center_inside(render_size, size)

        run_pipeline(loop, pipeline, {'appsink': {'new-sample':
            functools.partial(on_new_sample,
                render_overlay=functools.partial(render_overlay,
                    size=size,
                    window=window),
                render_size=render_size,
                images=images,
                commands=commands,
                fps_counter=avg_fps_counter(30))}
        })

    while GLib.MainContext.default().iteration(False):
        pass
