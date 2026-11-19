"""Design, inspect and use perceptual colormaps.

Run ``python examples/colormaps.py`` to write ``colormaps.png`` (needs matplotlib).
"""

import numpy as np

from lcms2.colormap import Colormap


def main() -> None:
    import matplotlib.pyplot as plt

    # One line: a colormap built from your own colours, interpolated in OKLab.
    custom = Colormap.from_colors(["#0b1d3a", "#2a9d8f", "#f4e285"], name="custom")
    maps = [
        Colormap.get("glacier"),
        Colormap.get("ember"),
        custom,
        Colormap.get("polar"),
        Colormap.get("halo"),
    ]

    for cm in maps:
        print(f"{cm.name:8s} monotonic L: {cm.is_lightness_monotonic()}  step CV: {cm.uniformity():.3f}")

    y, x = np.mgrid[-2:2:200j, -3:3:300j]
    data = np.sin(x * 2) * np.cos(y * 2) + 0.3 * x

    fig, axes = plt.subplots(2, len(maps), figsize=(3 * len(maps), 5), constrained_layout=True)
    for col, cm in enumerate(maps):
        axes[0, col].imshow(data, cmap=cm.to_matplotlib())  # two lines to use it in matplotlib
        axes[0, col].set_title(cm.name)
        # What the same map looks like with deuteranopia.
        axes[1, col].imshow(data, cmap=cm.simulate("deutan").to_matplotlib())
        axes[1, col].set_title("deutan")
    for ax in axes.ravel():
        ax.axis("off")
    fig.savefig("colormaps.png", dpi=120)
    print("wrote colormaps.png")


if __name__ == "__main__":
    main()
