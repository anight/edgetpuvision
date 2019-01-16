from collections import Sequence

def _clean(k):
    k = k[1:] if k.startswith('_') else k
    return k.replace('_', '-')

def rgb(color):
    return 'rgb(%s, %s, %s)' % color

def em(value):
    return '%sem' % value

def px(value):
    return '%spx' % value

def pt(value):
    return '%spt' % value

def mm(value):
    return '%smm' % value

def cm(value):
    return '%scm' % value

def inch(value):
    return '%sin' % value

class Style:
    def __init__(self, **attrs):
        self._attrs = attrs

    def __str__(self):
        return ';'.join('%s:%s' % (_clean(k), v) for k, v in self._attrs.items())

class Tag:
    NAME = None
    REQUIRED_ATTRS = ()

    def __init__(self, **attrs):
        self._attrs = attrs

        for attr in self.REQUIRED_ATTRS:
            if attr not in attrs:
                raise ValueError('Missing attribute "%s" from tag <%s/>' % (attr, self.NAME))

    @property
    def value(self):
        return None

    def __str__(self):
        sattrs = ' '.join('%s="%s"' % (_clean(k), v) for k, v in self._attrs.items())
        if sattrs:
            sattrs = ' ' + sattrs
        value = self.value
        if value is None:
            return '<%s%s/>' % (self.NAME, sattrs)
        return '<%s%s>%s</%s>' % (self.NAME, sattrs, value, self.NAME)

class TagContainer(Tag):
    def __init__(self, **attrs):
        super().__init__(**attrs)
        self._children = []

    def add(self, one_or_more):
        try:
            self._children.extend(one_or_more)
        except TypeError:
            self._children.append(one_or_more)

        return self

    def __iadd__(self, child):
        self.add(child)
        return self

    @property
    def value(self):
        return ''.join(str(child) for child in self._children)

class Svg(TagContainer):
    NAME = 'svg'

    def __init__(self, **attrs):
        super().__init__(**{'xmlns':'http://www.w3.org/2000/svg', **attrs})

class Group(TagContainer):
    NAME = 'g'

class Line(Tag):
    NAME = 'line'
    REQUIRED_ATTRS = ('x1', 'y1', 'x2', 'y2')

class Rect(Tag):
    NAME = 'rect'
    REQUIRED_ATTRS = ('x', 'y', 'width', 'height')

class Circle(Tag):
    NAME = 'circle'
    REQUIRED_ATTRS = ('cx', 'cy', 'r')

class Ellipse(Tag):
    NAME = 'ellipse'
    REQUIRED_ATTRS = ('cx', 'cy', 'rx', 'ry')

class Text(TagContainer):
    NAME = 'text'

    def __init__(self, text=None, **attrs):
        super().__init__(**attrs)
        self._text = text

    @property
    def value(self):
        if self._text:
            return self._text
        return super().value

class TSpan(Tag):
    NAME = 'tspan'

    def __init__(self, text, **attrs):
        super().__init__(**attrs)
        self._text = text

    @property
    def value(self):
        return self._text

class Path(Tag):
    NAME = 'path'
    REQUIRED_ATTRS = ('d',)

class Defs(TagContainer):
    NAME = 'defs'

class CssStyle(Tag):
    NAME = 'style'

    def __init__(self, styles):
        super().__init__(**{'_type': 'text/css'})
        self._styles = styles

    @property
    def value(self):
        return '<![CDATA[%s]]>' % '\n'.join('%s {%s}' % (k, v) for k, v in self._styles.items())



def shadow_text(arg, x, y, font_size_em=1.0, text_class='txt', shadow_class='shd'):
    lines = arg.split('\n') if isinstance(arg, str) else arg
    g = Group()
    if len(lines) == 1:
        g += Text(lines[0], x=x, y=y, dx=1, dy=1, _class=shadow_class, font_size=em(font_size_em))
        g += Text(lines[0], x=x, y=y, _class=text_class, font_size=em(font_size_em))
    elif len(lines) > 1:
        t = Text(y=y, dy=1, _class=shadow_class, font_size=em(font_size_em))
        for line in lines:
            t += TSpan(line, x=x, dx=1, dy=em(1.0))
        g += t

        t = Text(y=y, _class=text_class, font_size=em(font_size_em))
        for line in lines:
            t += TSpan(line, x=x, dy=em(1.0))
        g += t
    return g

def normal_text(arg, x, y, font_size_em=1.0, text_class='txt'):
    lines = arg.split('\n') if isinstance(arg, str) else arg

    if len(lines) == 1:
        return Text(lines[0], x=x, y=y, _class=text_class, font_size=em(font_size_em))
    elif len(lines) > 1:
        t = Text(y=y, _class='txt', font_size=em(font_size_em))
        for line in lines:
            t += TSpan(line, x=x, dy=em(1.0))
        return t
    else:
        raise ValueError('No text lines')
