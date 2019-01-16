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
import signal
import time

from edgetpu.detection.engine import DetectionEngine

from . import overlays
from .camera import InferenceCamera
from .streaming.server import StreamingServer
from .utils import load_labels

def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
    filtered_labels = set(x.strip() for x in args.filter.split(',')) if args.filter else None

    _, h, w, _ = engine.get_input_tensor_shape()

    camera = InferenceCamera((640, 360), (w, h))
    with StreamingServer(camera) as server:
        def on_image(rgb, inference_fps, size, view_box):
            start = time.monotonic()
            objs = engine.DetectWithInputTensor(rgb, threshold=args.threshold, top_k=args.top_k)
            inference_time = time.monotonic() - start

            if labels and filtered_labels:
                objs = [obj for obj in objs if labels[obj.label_id] in filtered_labels]

            server.send_overlay(overlays.detection(objs, inference_time, inference_fps, labels, size, view_box))

        camera.on_image = on_image
        signal.pause()

if __name__ == '__main__':
    main()
