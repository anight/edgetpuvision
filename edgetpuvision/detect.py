"""A demo which runs object detection on camera frames."""

# export TEST_DATA=/usr/lib/python3.5/dist-packages/edgetpu/test_data/
#
# Run face detection model:
# python3 detect.py \
#   --model=${TEST_DATA}/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite
#
# Run coco model:
# python3 detect.py \
#   --model=${TEST_DATA}/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite
#   --labels=${TEST_DATA}/coco_labels.txt

import argparse
import time

from edgetpu.detection.engine import DetectionEngine

from . import gstreamer
from . import overlays
from .utils import load_labels

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--source',
                        help='/dev/videoN:FMT:WxH:N/D or .mp4 file',
                        default='/dev/video0:YUY2:1280x720:30/1')
    parser.add_argument('--downscale', type=float, default=2.0,
                        help='Downscale factor for .mp4 file rendering.')
    parser.add_argument('--model',
                        help='.tflite model path.', required=True)
    parser.add_argument('--labels',
                        help='labels file path.')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Max number of objects to detect.')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='Detection threshold.')
    parser.add_argument('--filter', default=None)
    parser.add_argument('--fullscreen', default=False, action='store_true',
                        help='Fullscreen rendering.')
    args = parser.parse_args()

    engine = DetectionEngine(args.model)
    labels = load_labels(args.labels) if args.labels else None
    filtered_labels = set(x.strip() for x in args.filter.split(',')) if args.filter else None

    def render_overlay(rgb, size, view_box, inference_fps):
        start = time.monotonic()
        objs = engine.DetectWithInputTensor(rgb, threshold=args.threshold, top_k=args.top_k)
        inference_time  = time.monotonic() - start
        if labels and filtered_labels:
            objs = [obj for obj in objs if labels[obj.label_id] in filtered_labels]

        return overlays.detection(objs, inference_time, inference_fps, labels, size, view_box)

    _, h, w, _ = engine.get_input_tensor_shape()

    if not gstreamer.run((w, h), render_overlay,
                         source=args.source,
                         downscale=args.downscale,
                         fullscreen=args.fullscreen):
        print('Invalid source argument:', args.source)


if __name__ == '__main__':
    main()
