"""A demo which runs object detection and streams video to the browser."""

# export TEST_DATA=/usr/lib/python3.5/dist-packages/edgetpu/test_data/
#
# Run face detection model:
# python3 detect_server.py \
#   --model=${TEST_DATA}/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite
#
# Run coco model:
# python3 detect_server.py \
#   --model=${TEST_DATA}/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite
#   --labels=${TEST_DATA}/coco_labels.txt

import argparse
import logging
import os
import signal
import time

from edgetpu.detection.engine import DetectionEngine

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
    parser.add_argument('--model',
                        help='.tflite model path.', required=True)
    parser.add_argument('--labels',
                        help='labels file path.')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Max number of objects to detect.')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='Detection threshold.')
    parser.add_argument('--filter', default=None)
    args = parser.parse_args()

    engine = DetectionEngine(args.model)
    labels = load_labels(args.labels) if args.labels else None
    filtered_labels = set(l.strip() for l in args.filter.split(',')) if args.filter else None

    camera = make_camera(args.source, input_image_size(engine))
    assert camera is not None

    with StreamingServer(camera) as server:
        def on_image(tensor, inference_fps, size, window):
            start = time.monotonic()
            objs = engine.DetectWithInputTensor(tensor, threshold=args.threshold, top_k=args.top_k)
            inference_time = time.monotonic() - start

            if labels and filtered_labels:
                objs = [obj for obj in objs if labels[obj.label_id] in filtered_labels]

            server.send_overlay(overlays.detection(objs, labels, inference_time, inference_fps, size, window))

        camera.on_image = on_image
        signal.pause()

if __name__ == '__main__':
    main()
