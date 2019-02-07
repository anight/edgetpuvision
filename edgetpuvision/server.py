import argparse
import logging
import signal

from .camera import make_camera
from .streaming.server import StreamingServer

def run(add_render_gen_args, render_gen):
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--source',
                        help='/dev/videoN:FMT:WxH:N/D or .mp4 file or image file',
                        default='/dev/video0:YUY2:1280x720:30/1')
    add_render_gen_args(parser)
    args = parser.parse_args()

    gen = render_gen(args)
    camera = make_camera(args.source, next(gen))
    assert camera is not None

    with StreamingServer(camera) as server:
        def on_image(tensor, inference_rate, size, window):
            overlay = gen.send((tensor, size, window, inference_rate, None))
            server.send_overlay(overlay)

        camera.on_image = on_image
        signal.pause()
