import os

fn = os.path.join(os.path.abspath(os.path.dirname(__file__)), "illuminants.csv")

ILLUMINANTS = {2: dict(), 10: dict()}
skip_lines = 2
count = 0
with open(fn, "r") as f:
    for line in f:
        count += 1
        if count <= skip_lines:
            continue
        name, x2, y2, x10, y10, t, *description = line.strip().split(",")
        description = ",".join(description).replace("\"", "")
        ILLUMINANTS[2][name] = {"xyY": [float(x2), float(y2), 1.0],
                                "CCT": float(t),
                                "description": description}
        if len(x10) > 0:
            ILLUMINANTS[10][name] = {"xyY": [float(x10), float(y10), 1.0],
                                    "CCT": float(t),
                                    "description": description}

print(ILLUMINANTS)