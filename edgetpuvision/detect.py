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
import collections
import colorsys
import itertools
import time

from edgetpu.detection.engine import DetectionEngine

from . import svg
from .apps import run_app
from .utils import load_labels, input_image_size, same_input_image_sizes, avg_fps_counter

CSS_STYLES = str(svg.CssStyle({'.txt': svg.Style(fill='white'),
                               '.shd': svg.Style(fill='black', fill_opacity=0.6),
                               'rect': svg.Style(fill_opacity=0.1, stroke_width='2px')}))

BBox = collections.namedtuple('BBox', ('x', 'y', 'w', 'h'))
BBox.area = lambda self: self.w * self.h
BBox.scale = lambda self, sx, sy: BBox(x=self.x * sx, y=self.y * sy,
                                       w=self.w * sx, h=self.h * sy)
BBox.__str__ = lambda self: 'BBox(x=%.2f y=%.2f w=%.2f h=%.2f)' % self

Object = collections.namedtuple('Object', ('id', 'label', 'score', 'bbox'))
Object.__str__ = lambda self: 'Object(id=%d, label=%s, score=%.2f, %s)' % self

def color(i, total):
    return tuple(int(255.0 * c) for c in colorsys.hsv_to_rgb(i / total, 1.0, 1.0))

def gen_colors_unique(keys):
    return {key : svg.rgb(color(i, len(keys))) for i, key in enumerate(keys)}

def gen_colors_single(keys, color):
    return {key : color for key in keys}

def overlay(objs, colors, inference_time, inference_rate, layout):
    x0, y0, w, h = layout.window

    defs = svg.Defs()
    defs += CSS_STYLES

    doc = svg.Svg(width=w, height=h, viewBox='%s %s %s %s' % layout.window, font_size='26px')
    doc += defs
    doc += svg.normal_text((
        'Inference time: %.2f ms (%.2f fps)' % (inference_time * 1000, 1.0 / inference_time),
        'Inference frame rate: %.2f fps' % inference_rate,
        'Objects: %d' % len(objs),
    ), x0 + 10, y0 + 10, font_size_em=1.1)

    for obj in objs:
        percent = int(100 * obj.score)
        if obj.label:
            caption = '%d%% %s' % (percent, obj.label)
        else:
            caption = '%d%%' % percent

        x, y, w, h = obj.bbox.scale(*layout.size)
        doc += svg.normal_text(caption, x, y - 5)
        doc += svg.Rect(x=x, y=y, width=w, height=h, rx=2, ry=2,
                        style='stroke:%s' % colors[obj.id])

    return str(doc)


def convert(obj, labels):
    x0, y0, x1, y1 = obj.bounding_box.flatten().tolist()
    return Object(id=obj.label_id,
                  label=labels[obj.label_id] if labels else None,
                  score=obj.score,
                  bbox=BBox(x=x0, y=y0, w=x1 - x0, h=y1 - y0))

def print_results(inference_rate, objs):
    print('\nInference (rate=%.2f fps):' % inference_rate)
    for i, obj in enumerate(objs):
        print('    %d: %s, area=%.2f' % (i, obj, obj.bbox.area()))

def render_gen(args):
    fps_counter=avg_fps_counter(30)

    engines = [DetectionEngine(m) for m in args.model.split(',')]
    assert same_input_image_sizes(engines)
    engines = itertools.cycle(engines)
    engine = next(engines)

    labels = load_labels(args.labels) if args.labels else None
    filtered_labels = set(l.strip() for l in args.filter.split(',')) if args.filter else None
    if args.color:
        colors = gen_colors_single(labels.keys(), args.color)
    else:
        colors = gen_colors_unique(labels.keys())
    draw_overlay = True

    yield input_image_size(engine)

    output = None
    while True:
        tensor, layout, command = (yield output)

        inference_rate = next(fps_counter)
        if draw_overlay:
            start = time.monotonic()
            objs = engine.DetectWithInputTensor(tensor, threshold=args.threshold, top_k=args.top_k)
            inference_time = time.monotonic() - start
            objs = [convert(obj, labels) for obj in objs]

            if labels and filtered_labels:
                objs = [obj for obj in objs if obj.label in filtered_labels]

            objs = [obj for obj in objs if args.min_area <= obj.bbox.area() <= args.max_area]

            if args.print:
                print_results(inference_rate, objs)

            output = overlay(objs, colors, inference_time, inference_rate, layout)
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
    parser.add_argument('--color', default=None,
                        help='Bounding box display color'),
    parser.add_argument('--print', default=False, action='store_true',
                        help='Print inference results')

def main():
    run_app(add_render_gen_args, render_gen)

if __name__ == '__main__':
    main()
