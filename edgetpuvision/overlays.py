from . import svg

CSS_STYLES = str(svg.CssStyle({'.txt': svg.Style(fill='white'),
                               '.shd': svg.Style(fill='black', fill_opacity=0.6),
                               'rect': svg.Style(fill='green', fill_opacity=0.3, stroke='white')}))


def _normalize_rect(rect, size):
    width, height = size
    x0, y0, x1, y1 = rect
    return int(x0 * width), int(y0 * height), \
           int((x1 - x0) * width), int((y1 - y0) * height)


def classification(results, inference_time, inference_rate, layout):
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

def detection(objs, labels, inference_time, inference_rate, layout):
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
        if labels:
            caption = '%d%% %s' % (percent, labels[obj.label_id])
        else:
            caption = '%d%%' % percent

        x, y, w, h = _normalize_rect(obj.bounding_box.flatten().tolist(), layout.size)
        doc += svg.normal_text(caption, x, y - 5)
        doc += svg.Rect(x=x, y=y, width=w, height=h, rx=2, ry=2)

    return str(doc)
