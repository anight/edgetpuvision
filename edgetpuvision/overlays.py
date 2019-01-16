from . import svg

CSS_STYLES = str(svg.CssStyle({'.txt': svg.Style(fill='white'),
                               '.shd': svg.Style(fill='black', fill_opacity=0.6),
                               'rect': svg.Style(fill='green', fill_opacity=0.3, stroke='white')}))


def _normalize_rect(rect, size):
    width, height = size
    x0, y0, x1, y1 = rect
    x, y, w, h = x0, y0, x1 - x0, y1 - y0
    return int(x * width), int(y * height), int(w * width), int(h * height)


def classification(results, inference_time, inference_fps, size, view_box):
    x0, y0, _, _ = view_box

    lines = [
        'Inference time: %.2f ms (%.2f fps)' % (inference_time * 1000, 1.0 / inference_time),
        'Inference frame rate: %.2f fps' % inference_fps
    ]

    for i, (label, score) in enumerate(results):
        lines.append('%s (%.2f)' % (label, score))

    defs = svg.Defs()
    defs += CSS_STYLES

    doc = svg.Svg(viewBox='%s %s %s %s' % view_box, font_size='26px')
    doc += defs
    doc += svg.normal_text(lines, x=x0 + 10, y=y0 + 10, font_size_em=1.1)
    return str(doc)


def detection(objs, inference_time, inference_fps, labels, size, view_box):
    x0, y0, _, _ = view_box

    defs = svg.Defs()
    defs += CSS_STYLES

    doc = svg.Svg(viewBox='%s %s %s %s' % view_box, font_size='26px')
    doc += defs
    doc += svg.normal_text((
        'Inference time: %.2f ms (%.2f fps)' % (inference_time * 1000, 1.0 / inference_time),
        'Inference frame rate: %.2f fps' % inference_fps,
        'Objects: %d' % len(objs),
    ), x0 + 10, y0 + 10, font_size_em=1.1)

    for obj in objs:
        percent = int(100 * obj.score)
        if labels:
            label = labels[obj.label_id]
            caption = '%d%% %s' % (percent, label)
        else:
            caption = '%d%%' % percent

        x, y, w, h = _normalize_rect(obj.bounding_box.flatten().tolist(), size)
        doc += svg.normal_text(caption, x, y - 5)
        doc += svg.Rect(x=x, y=y, width=w, height=h, rx=2, ry=2)

    return str(doc)
