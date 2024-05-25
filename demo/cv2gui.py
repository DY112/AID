import sys
from collections import defaultdict
import cv2
import numpy as np

class Window:
    def __init__(self, name, width=640, height=480, background_color=(0, 0, 0)):
        self.name = name
        self.width = width
        self.height = height
        self.background_color = background_color
        self.objects = []
        self.canvas = np.zeros((self.height, self.width, 3), np.uint8)
        self.fps = 60
        self.event_listeners = defaultdict(list)

        cv2.namedWindow(self.name)
        cv2.resizeWindow(self.name, self.width, self.height)
        cv2.setMouseCallback(self.name, self.mouse_callback)

    def show(self):
        while self:
            self.update()
            key = cv2.waitKey(1 if self.fps < 0 else int(1000 / self.fps))
            if key != -1:
                for callback in self.event_listeners['onKeyDown']:
                    callback(key)
                if key == 27: # ESC
                    self.close()
                    break

    def update(self):
        for callback in self.event_listeners['onUpdate']:
            callback(self.delta)
        self.draw()
        cv2.imshow(self.name, self.canvas)

    def draw(self):
        self.canvas[:] = self.background_color
        for obj in self.objects:
            # print(sys.getrefcount(obj))
            obj.draw(self.canvas)

    def add(self, obj):
        self.objects.append(obj)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for callback in self.event_listeners['onMouseDown']:
                callback(x, y)
        if event == cv2.EVENT_MOUSEMOVE:
            for callback in self.event_listeners['onMouseMove']:
                callback(x, y)
        if event == cv2.EVENT_LBUTTONUP:
            for callback in self.event_listeners['onMouseUp']:
                callback(x, y)

    def add_event_listener(self, event, callback):
        self.event_listeners[event].append(callback)

    def close(self):
        try:
            cv2.destroyWindow(self.name)
        except:
            pass

    @property
    def delta(self):
        return 1 / self.fps

    def __bool__(self):
        return cv2.getWindowProperty(self.name, cv2.WND_PROP_VISIBLE) > 0
    
    def __nonzero__(self):
        return self.__bool__()

    def __del__(self):
        self.close()

class Object:
    def __init__(self, x=0, y=0):
        self._x = x
        self._y = y
        self.childs = []
        self.visible = True

    def draw(self, canvas):
        if not self.visible:
            return
        for obj in self.childs:
            obj.draw(canvas)

    def add(self, obj):
        self.childs.append(obj)

    @property
    def x(self):
        return self._x
    
    @x.setter
    def x(self, value):
        dx = value - self._x
        self._x += dx
        for obj in self.childs:
            obj.x += dx

    @property
    def y(self):
        return self._y
    
    @y.setter
    def y(self, value):
        dy = value - self._y
        self._y += dy
        for obj in self.childs:
            obj.y += dy

    @property
    def pos(self):
        return (self.x, self.y)
    
    @pos.setter
    def pos(self, value):
        self.x, self.y = value
    

class Sprite(Object):
    def __init__(self, data, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if type(data) is str:
            data = cv2.imread(data)
        self.pixels = np.array(data, dtype=np.uint8)

    def draw(self, canvas):
        if not self.visible:
            return
        sx = max(-self.x, 0)
        sy = max(-self.y, 0)
        ex = self.pixels.shape[1] - max(self.x + self.pixels.shape[1] - canvas.shape[1], 0)
        ey = self.pixels.shape[0] - max(self.y + self.pixels.shape[0] - canvas.shape[0], 0)
        if sx < ex and sy < ey:
            if self.pixels.shape[-1] == 4: # alpha blending
                alpha = self.pixels[sy:ey, sx:ex, 3:4] / 255
                area = canvas[self.y+sy:self.y+ey, self.x+sx:self.x+ex]
                canvas[self.y+sy:self.y+ey, self.x+sx:self.x+ex] = (1 - alpha) * area + alpha * self.pixels[sy:ey, sx:ex, :3]
            else:
                canvas[self.y+sy:self.y+ey, self.x+sx:self.x+ex] = self.pixels[sy:ey, sx:ex]
        super().draw(canvas)

    def scale(self, factor):
        self.pixels = cv2.resize(self.pixels, (0, 0), fx=factor, fy=factor)

    def contain(self, x, y):
        return 0 <= x - self.x < self.width and 0 <= y - self.y < self.height
    
    @property
    def width(self):
        return self.pixels.shape[1]
    
    @property
    def height(self):
        return self.pixels.shape[0]
    
class Text(Object):
    def __init__(self, text, *args, font_size=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.text = text
        self.font = cv2.FONT_HERSHEY_COMPLEX
        self.font_scale = font_size / 50
        self.font_color = (255, 255, 255)
        self.font_thickness = 1
        self.font_line_type = cv2.LINE_AA
    
    def draw(self, canvas):
        if not self.visible:
            return
        textsize = cv2.getTextSize(self.text, self.font, self.font_scale, self.font_thickness)[0]
        cv2.putText(canvas, self.text, (self.x-textsize[0]//2, self.y), self.font, self.font_scale, self.font_color, self.font_thickness, self.font_line_type)
        super().draw(canvas)
