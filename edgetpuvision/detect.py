"""A demo which runs object detection on camera frames."""

# export TEST_DATA=/usr/lib/python3/dist-packages/edgetpu/test_data
#
# Run face detection model:
# python3 -m edgetpuvision.detect \
#   --model ${TEST_DATA}/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite
#
# Run coco model:
# python3 -m edgetpuvision.detect \
#   --model ${TEST_DATA}/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
#   --labels ${TEST_DATA}/coco_labels.txt

import argparse
import itertools
import time

from edgetpu.detection.engine import DetectionEngine


from . import overlays
from .utils import load_labels, input_image_size, same_input_image_sizes
from .gstreamer import Display, run_gen

def area(obj):
    x0, y0, x1, y1 = rect = obj.bounding_box.flatten().tolist()
    return (x1 - x0) * (y1 - y0)

def print_results(inference_rate, objs, labels):
    print('\nInference (rate=%.2f fps):' % inference_rate)
    for i, obj in enumerate(objs):
        label = labels[obj.label_id] if labels else str(obj.label_id)
        x = (i, label) + tuple(obj.bounding_box.flatten()) + (area(obj),)
        print('    %d: label=%s, bbox=(%.2f %.2f %.2f %.2f), bbox_area=%.2f' % x)

def render_gen(args):
    engines = [DetectionEngine(m) for m in args.model.split(',')]
    assert same_input_image_sizes(engines)
    engines = itertools.cycle(engines)
    engine = next(engines)

    labels = load_labels(args.labels) if args.labels else None
    filtered_labels = set(l.strip() for l in args.filter.split(',')) if args.filter else None
    draw_overlay = True

    yield input_image_size(engine)

    output = None
    while True:
        tensor, size, window, inference_rate, command = (yield output)
        if draw_overlay:
            start = time.monotonic()
            objs = engine.DetectWithInputTensor(tensor, threshold=args.threshold, top_k=args.top_k)
            inference_time = time.monotonic() - start

            if labels and filtered_labels:
                objs = [obj for obj in objs if labels[obj.label_id] in filtered_labels]

            objs = [obj for obj in objs if args.min_area <= area(obj) <= args.max_area]

            if args.print:
                print_results(inference_rate, objs, labels)

            output = overlays.detection(objs, labels, inference_time, inference_rate, size, window)
        else:
            output = None

        if command == 'o':
            draw_overlay = not draw_overlay
        elif command == 'n':
            engine = next(engines)

def add_render_gen_args(parser):
    parser.add_argument('--model',
                        help='.tflite model path', required=True)
    parser.add_argument('--labels',
                        help='labels file path')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Max number of objects to detect')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='Detection threshold')
    parser.add_argument('--min_area', type=float, default=0.0,
                        help='Min bounding box area')
    parser.add_argument('--max_area', type=float, default=1.0,
                        help='Max bounding box area')
    parser.add_argument('--filter', default=None,
                        help='Comma-separated list of allowed labels')
    parser.add_argument('--print', default=False, action='store_true',
                        help='Print inference results')

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--source',
                        help='/dev/videoN:FMT:WxH:N/D or .mp4 file or image file',
                        default='/dev/video0:YUY2:1280x720:30/1')
    parser.add_argument('--downscale', type=float, default=2.0,
                        help='Downscale factor for video/image file rendering')
    parser.add_argument('--display', type=Display, choices=Display, default=Display.FULLSCREEN,
                        help='Display mode')
    add_render_gen_args(parser)
    args = parser.parse_args()

    if not run_gen(render_gen(args),
                   source=args.source,
                   downscale=args.downscale,
                   display=args.display):
        print('Invalid source argument:', args.source)


if __name__ == '__main__':
    main()
