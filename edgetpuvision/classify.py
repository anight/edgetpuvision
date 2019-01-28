"""A demo which runs object classification on camera frames."""

# export TEST_DATA=/usr/lib/python3/dist-packages/edgetpu/test_data
#
# python3 -m edgetpuvision.classify \
#   --model ${TEST_DATA}/mobilenet_v2_1.0_224_inat_bird_quant.tflite \
#   --labels ${TEST_DATA}/inat_bird_labels.txt

import argparse
import collections
import itertools
import time

from edgetpu.classification.engine import ClassificationEngine

from . import gstreamer
from . import overlays
from .utils import load_labels, input_image_size, same_input_image_sizes


def top_results(window, top_k):
    total_scores = collections.defaultdict(lambda: 0.0)
    for results in window:
        for label, score in results:
            total_scores[label] += score
    return sorted(total_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]


def accumulator(size, top_k):
    window = collections.deque(maxlen=size)
    window.append((yield []))
    while True:
        window.append((yield top_results(window, top_k)))


def render_gen(args):
    acc = accumulator(size=args.window, top_k=args.top_k)
    acc.send(None)  # Initialize.

    engines = [ClassificationEngine(m) for m in args.model.split(',')]
    assert same_input_image_sizes(engines)
    engines = itertools.cycle(engines)
    engine = next(engines)

    labels = load_labels(args.labels)
    draw_overlay = True

    yield input_image_size(engine)

    output = None
    while True:
        tensor, size, window, inference_rate, command = (yield output)

        if draw_overlay:
            start = time.monotonic()
            results = engine.ClassifyWithInputTensor(tensor, threshold=args.threshold, top_k=args.top_k)
            inference_time = time.monotonic() - start

            results = [(labels[i], score) for i, score in results]
            results = acc.send(results)
            if args.print:
                print(results)

            output = overlays.classification(results, inference_time, inference_rate, size, window)
        else:
            output = None

        if command == 'o':
            draw_overlay = not draw_overlay
        elif command == 'n':
            engine = next(engines)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--source',
                        help='/dev/videoN:FMT:WxH:N/D or .mp4 file or image file',
                        default='/dev/video0:YUY2:1280x720:30/1')
    parser.add_argument('--downscale', type=float, default=2.0,
                        help='Downscale factor for .mp4 file rendering')
    parser.add_argument('--model', required=True,
                        help='.tflite model path')
    parser.add_argument('--labels', required=True,
                        help='label file path')
    parser.add_argument('--window', type=int, default=10,
                        help='number of frames to accumulate inference results')
    parser.add_argument('--top_k', type=int, default=3,
                        help='number of classes with highest score to display')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='class score threshold')
    parser.add_argument('--print', action='store_true', default=False,
                        help='Print detected classes to console')
    parser.add_argument('--fullscreen', default=False, action='store_true',
                        help='Fullscreen rendering')
    args = parser.parse_args()

    if not gstreamer.run_gen(render_gen(args),
                         source=args.source,
                         downscale=args.downscale,
                         fullscreen=args.fullscreen):
        print('Invalid source argument:', args.source)

if __name__ == '__main__':
    main()
