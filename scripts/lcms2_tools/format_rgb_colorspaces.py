import os

fn = os.path.join(os.path.abspath(os.path.dirname(__file__)), "rgb_color_spaces.csv")

RGB_COLORSPACES = dict()
skip_lines = 7
count = 0
with open(fn, "r") as f:
    for line in f:
        count += 1
        if count <= skip_lines:
            continue
        name, gamma, wp, rx, ry, rY, gx, gy, gY, bx, by, bY, *rest = line.strip().split(",")
        tone_curve = dict()
        try:
            gamma = float(gamma)
            tone_curve = {"type": 1, "parameters": [gamma]}
        except ValueError:
            pass
        RGB_COLORSPACES[name] = {"tone curve": tone_curve,
                                "white point": wp,
                                "red": [float(v) for v in (rx, ry, rY)],
                                "green": [float(v) for v in (gx, gy, gY)],
                                 "blue": [float(v) for v in (bx, by, bY)]}

print(RGB_COLORSPACES)