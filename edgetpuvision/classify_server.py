"""A demo which runs object classification and streams video to the browser."""

#export TEST_DATA=/usr/lib/python3.5/dist-packages/edgetpu/test_data/
#
# python3 classify_server.py \
#   --model=${TEST_DATA}/mobilenet_v1_1.0_224_quant_edgetpu.tflite \
#   --labels=${TEST_DATA}/imagenet_labels.txt

import argparse
import logging
import signal
import time

from edgetpu.classification.engine import ClassificationEngine

from . import overlays
from .camera import make_camera
from .streaming.server import StreamingServer
from .utils import load_labels, input_image_size


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--source',
                        help='/dev/videoN:FMT:WxH:N/D or .mp4 file',
                        default='/dev/video0:YUY2:1280x720:30/1')
    parser.add_argument('--model', required=True,
                        help='.tflite model path.')
    parser.add_argument('--labels', required=True,
                        help='label file path.')
    parser.add_argument('--top_k', type=int, default=3,
                        help='number of classes with highest score to display.')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='class score threshold.')
    args = parser.parse_args()

    engine = ClassificationEngine(args.model)
    labels = load_labels(args.labels)

    camera = make_camera(args.source, input_image_size(engine))
    assert camera is not None

    with StreamingServer(camera) as server:
        def on_image(tensor, inference_fps, size, window):
            start = time.monotonic()
            results = engine.ClassifyWithInputTensor(tensor, threshold=args.threshold, top_k=args.top_k)
            inference_time = time.monotonic() - start

            results = [(labels[i], score) for i, score in results]
            server.send_overlay(overlays.classification(results, inference_time, inference_fps, size, window))

        camera.on_image = on_image
        signal.pause()

if __name__ == '__main__':
    main()
