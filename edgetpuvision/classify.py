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

from . import svg
from .apps import run_app
from .utils import load_labels, input_image_size, same_input_image_sizes, avg_fps_counter


CSS_STYLES = str(svg.CssStyle({'.txt': svg.Style(fill='white'),
                               '.shd': svg.Style(fill='black', fill_opacity=0.6)}))

def overlay(results, inference_time, inference_rate, layout):
    x0, y0, w, h = layout.window

    lines = [
        'Inference time: %.2f ms (%.2f fps)' % (inference_time * 1000, 1.0 / inference_time),
        'Inference frame rate: %.2f fps' % inference_rate
    ]

    for i, (label, score) in enumerate(results):
        lines.append('%s (%.2f)' % (label, score))

    defs = svg.Defs()
    defs += CSS_STYLES

    doc = svg.Svg(width=w, height=h, viewBox='%s %s %s %s' % layout.window, font_size='26px')
    doc += defs
    doc += svg.normal_text(lines, x=x0 + 10, y=y0 + 10, font_size_em=1.1)
    return str(doc)

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

def print_results(inference_rate, results):
    print('\nInference (rate=%.2f fps):' % inference_rate)
    print(results)
    for label, score in results:
        print('  %s, score=%.2f' % (label, score))

def render_gen(args):
    acc = accumulator(size=args.window, top_k=args.top_k)
    acc.send(None)  # Initialize.

    fps_counter=avg_fps_counter(30)

    engines = [ClassificationEngine(m) for m in args.model.split(',')]
    assert same_input_image_sizes(engines)
    engines = itertools.cycle(engines)
    engine = next(engines)

    labels = load_labels(args.labels)
    draw_overlay = True

    yield input_image_size(engine)

    output = None
    while True:
        tensor, layout, command = (yield output)

        inference_rate = next(fps_counter)
        if draw_overlay:
            start = time.monotonic()
            results = engine.ClassifyWithInputTensor(tensor, threshold=args.threshold, top_k=args.top_k)
            inference_time = time.monotonic() - start

            results = [(labels[i], score) for i, score in results]
            results = acc.send(results)
            if args.print:
                print_results(inference_rate, results)

            output = overlay(results, inference_time, inference_rate, layout)
        else:
            output = None

        if command == 'o':
            draw_overlay = not draw_overlay
        elif command == 'n':
            engine = next(engines)

def add_render_gen_args(parser):
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
    parser.add_argument('--print', default=False, action='store_true',
                        help='Print inference results')

def main():
    run_app(add_render_gen_args, render_gen)

if __name__ == '__main__':
    main()
