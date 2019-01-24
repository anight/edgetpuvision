import re

LABEL_PATTERN = re.compile(r'\s*(\d+)(.+)')

def load_labels(path):
    with open(path, 'r', encoding='utf-8') as f:
       lines = (LABEL_PATTERN.match(line).groups() for line in f.readlines())
       return {int(num): text.strip() for num, text in lines}


def input_image_size(engine):
    _, h, w, _ = engine.get_input_tensor_shape()
    return w, h

def same_input_image_sizes(engines):
    return len({input_image_size(engine) for engine in engines}) == 1
