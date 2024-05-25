from aid import AIDResult
import cv2gui as cg
import numpy as np
import cv2

UNIT = 256
GAP = 50
FONT_SIZE = 35
PAD = 100
GRID_WIDTH = UNIT + GAP
GRID_HEIGHT = UNIT + GAP
WIN_WIDTH = GRID_WIDTH * 3 + PAD * 2
WIN_HEIGHT = GRID_HEIGHT * 2 + PAD * 2
ROOT_PATH = "result"

class ColorPicker(cg.Sprite):
    
    def __init__(self, size, *args, ratio=3, **kwargs):
        self.ratio = ratio
        super().__init__(self.gen_color_space(size), *args, **kwargs)
        self.gen_circle(size)
        self.set_by_color([1.0, 1.0])

    def set_by_pos(self, x, y):
        self.tint[0] = x / self.height * self.ratio # R
        self.tint[1] = y / self.width * self.ratio # B
        self.update_circle_pos(x, y)

    def set_by_color(self, tint):
        self.tint = np.array(tint) # [R, B]
        x = int(self.tint[0] / self.ratio * self.height) # R
        y = int(self.tint[1] / self.ratio * self.width) # B
        self.update_circle_pos(x, y)

    def gen_color_space(self, size):
        color_space = np.zeros((size, size, 3))
        y, x = np.indices((size, size))
        b, r = y / size * self.ratio, x / size * self.ratio
        g = np.ones_like(r) * 1
        color_space[..., 0] = b
        color_space[..., 1] = g
        color_space[..., 2] = r
        color_space /= color_space.max(axis=2, keepdims=True)
        color_space = (color_space * 255).astype(np.uint8)
        return color_space

    def gen_circle(self, size):
        radius = size//20
        border = 1 # int(size//100)
        circle = np.zeros((radius*2, radius*2, 4), dtype=np.uint8)
        cv2.circle(circle, (radius, radius), radius-border, (255, 255, 255, 255), border, lineType=cv2.LINE_AA)
        self.circle = cg.Sprite(circle)
        self.add(self.circle)

    def update_circle_pos(self, x, y):
        self.circle.x = self.x + x - self.circle.width//2
        self.circle.y = self.y + y - self.circle.height//2

if __name__ == "__main__":
    print("AID Demo")
    print("Usage:")
    print("- Press 'n' to load next scene")
    print("- Press 'p' to load previous scene")
    print("- Press 'r' to reset color picker")
    print("- Press esc to quit")
    print("ROOT_PATH:", ROOT_PATH)
    print()

    win = cg.Window("AID Demo", WIN_WIDTH, WIN_HEIGHT)

    # Container
    container = cg.Object(x=0, y=0)
    win.add(container)

    # Sprites
    once = np.ones((UNIT, UNIT, 3), dtype=np.uint8) * 255
    sprite_rgb_in = cg.Sprite(once, x=0, y=0)
    sprite_grid_out = cg.Sprite(once, x=GRID_WIDTH, y=0)
    # sprite_rgb_wb = cg.Sprite(once, x=0, y=GRID_HEIGHT)
    sprite_raw_tint = cg.Sprite(once, x=0, y=GRID_HEIGHT)
    sprite_rgb_tint = cg.Sprite(once, x=GRID_WIDTH, y=GRID_HEIGHT)

    container.add(sprite_rgb_in)
    container.add(sprite_grid_out)
    # container.add(sprite_rgb_wb)
    container.add(sprite_raw_tint)
    container.add(sprite_rgb_tint)

    # Color Picker
    color_picker1 = ColorPicker(UNIT//2, x=GRID_WIDTH*2, y=GRID_HEIGHT+FONT_SIZE)
    color_picker2 = ColorPicker(UNIT//2, x=GRID_WIDTH*2 + GRID_WIDTH//2, y=GRID_HEIGHT+FONT_SIZE)

    container.add(color_picker1)
    container.add(color_picker2)

    # Labels
    label_dict = {
        "Ref. JPG": sprite_rgb_in,
        "Output": sprite_grid_out,
        # "rgb_wb": sprite_rgb_wb,
        "RAW": sprite_raw_tint,
        "sRGB": sprite_rgb_tint,
        "Illum. 1": color_picker1,
        "Illum. 2": color_picker2,
    }
    labels = {}
    for name, sprite in label_dict.items():
        label = cg.Text(name, font_size=FONT_SIZE)
        label.x = sprite.x + sprite.width//2
        label.y = sprite.y + sprite.height + GAP//2 + 4
        labels[name] = label
        container.add(label)

    def Tint2Text(tint):
        return "R: {:.2f}, G: 1.0, B: {:.2f}".format(tint[0], tint[1])
    label_tint1 = cg.Text("Tint 1", font_size=FONT_SIZE//2, x=color_picker1.x + color_picker1.width//2, y=color_picker1.y - 10)
    label_tint2 = cg.Text("Tint 2", font_size=FONT_SIZE//2, x=color_picker2.x + color_picker2.width//2, y=color_picker2.y - 10)
    container.add(label_tint1)
    container.add(label_tint2)

    container.pos = (PAD+15, PAD+15)

    # Load Result
    result = None
    # tint1, tint2 = None, None
    tints = None

    def load_result(name):
        print("Load Result: {}".format(name))
        global result
        global tints
        # global tint1, tint2
        result = AIDResult(ROOT_PATH, name)
        tints = np.array(result.chromas) # [R, B]
        # tint1 = np.array(result.chroma1) # [R, B]
        # tint2 = np.array(result.chroma2) # [R, B]
        rgb_in = result.rgb_in # BGR
        grid_out = result.grid_out # BGR
        # rgb_wb = cv2.cvtColor(result.rgb_wb, cv2.COLOR_RGB2BGR) # RGB => BGR

        sprite_rgb_in.pixels = rgb_in
        sprite_grid_out.pixels = grid_out
        # sprite_grid_out.scale(UNIT / grid_out.shape[0], UNIT / grid_out.shape[1])
        # sprite_rgb_wb.pixels = rgb_wb

        color_picker1.set_by_color(tints[0]) # [R, B]
        color_picker2.visible = False
        labels['Illum. 2'].visible = False
        if len(tints) > 1:
            color_picker2.set_by_color(tints[1])
            color_picker2.visible = True
            labels['Illum. 2'].visible = True
        # color_picker2.set_by_color(tint2) # [R, B]

        update_images()

    def update_images():
        global tints
        # global tint1, tint2
        tint1 = color_picker1.tint # [R, B]
        tint2 = color_picker2.tint # [R, B]
        if len(tints) > 1:
            label_tint1.text = Tint2Text(tint1)
            label_tint2.text = Tint2Text(tint2)
            label_tint2.visible = True
            tints = np.array([tint1, tint2])
        else:
            label_tint1.text = Tint2Text(tint1)
            label_tint2.visible = False
            tints = np.array([tint1])
        print("tint(R,B): [{:.2f}, {:.2f}] [{:.2f}, {:.2f}] \r".format(
                tint1[0], tint1[1], tint2[0], tint2[1]
            ), end="")
        if len(tints) > 1:
            raw, rgb = result.tinted(tint1, tint2)
        else:
            raw, rgb = result.tinted(tint1)
        sprite_raw_tint.pixels = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
        sprite_rgb_tint.pixels = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if len(tints) > 1:
            sprite_grid_out.x = container.x + GRID_WIDTH
        else:
            sprite_grid_out.x = container.x + GRID_WIDTH + UNIT//4
            

    scene_list = open("list.txt", "r").read().strip().splitlines()
    scene_idx = 0
    load_result(scene_list[scene_idx])

    # Mouse Event
    picking1 = False
    picking2 = False

    def set_color_picker(x, y, color_picker):
        global tints
        # global tint1, tint2
        if color_picker.contain(x, y):
            color_picker.set_by_pos(x-color_picker.x, y-color_picker.y)
            update_images()

    def on_mouse_down(x, y):
        global picking1, picking2
        if color_picker1.contain(x, y):
            picking1 = True
            set_color_picker(x, y, color_picker1)
        if color_picker2.contain(x, y):
            picking2 = True
            set_color_picker(x, y, color_picker2)

    def on_mouse_move(x, y):
        if (picking1):
            set_color_picker(x, y, color_picker1)
        if (picking2):
            set_color_picker(x, y, color_picker2)

    def on_mouse_up(x, y):
        global picking1, picking2
        picking1 = False
        picking2 = False

    win.add_event_listener("onMouseDown", on_mouse_down)
    win.add_event_listener("onMouseMove", on_mouse_move)
    win.add_event_listener("onMouseUp", on_mouse_up)

    # Keyboard Event
    def on_key_down(key):
        global scene_idx
        if key == ord('r'):
            print("\nReset")
            color_picker1.set_by_color(result.chroma1)
            color_picker2.set_by_color(result.chroma2)
            update_images()
        if key == ord('n'):
            print("\nNext")
            scene_idx = (scene_idx + 1) % len(scene_list)
            load_result(scene_list[scene_idx])
        if key == ord('p'):
            print("\nPrevious")
            scene_idx = (scene_idx - 1) % len(scene_list)
            load_result(scene_list[scene_idx])

    win.add_event_listener("onKeyDown", on_key_down)

    # Main Loop Start
    win.show()
