"""Three-panel figure: why a Gaussian blur commutes with a convolution, and
where that shortcut stops working inside a network.

Panel A -- the identity  G_s * (K * H) = (G_s * K) * H, drawn with real
arrays: both rows are computed by FFT circular convolution, so the equals sign
between the two output images is a measured fact (residual printed below the
figure and on stdout), not an assertion.

Panel B -- the ReLU roadblock.  The blur slides backwards through Conv 2 but
not through the pointwise non-linearity; the mismatch is measured on the same
arrays and reported as a relative L2 difference.

Panel C -- the striding caveat.  Three feature maps drawn at a *constant cell
pitch*, so the grids shrink while a sigma = 2 px disc keeps its physical size:
the same blur covers a growing fraction of the map as the map is downsampled.

Writes results/presentation/blur_commutation.{png,pdf,svg}.  No training, no
data dependency: every array in the figure is generated here from a fixed seed.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "presentation"
NAME = "blur_commutation"

N = 32                 # feature-map side used in panels A and B
SIGMA_A = 1.6          # blur width for the identity panel, in map pixels
SIGMA_C = 2.0          # blur width advertised in the striding panel
SEED = 0

RED = "#d62728"
BLUE = "#1f77b4"
GREEN = "#2ca02c"
GREY = "#7a7a7a"

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11,
                     "axes.labelsize": 10, "legend.fontsize": 9,
                     "figure.facecolor": "white"})


# --------------------------------------------------------------------------
# exact circular convolution: kernels live as image-sized fields whose centre
# sits at index (0, 0), so every product below is exact up to float round-off.
# --------------------------------------------------------------------------
def as_field(kernel, n=N):
    """Embed a small centred kernel into an n x n field with origin at (0,0)."""
    kh, kw = kernel.shape
    f = np.zeros((n, n))
    f[:kh, :kw] = kernel
    return np.roll(f, (-(kh // 2), -(kw // 2)), axis=(0, 1))


def gaussian_field(sigma, n=N):
    """Periodic Gaussian of width sigma on the n x n torus, summing to one."""
    d = np.minimum(np.arange(n), n - np.arange(n))
    g1 = np.exp(-0.5 * (d / sigma) ** 2)
    g = np.outer(g1, g1)
    return g / g.sum()


def conv(a, field):
    return np.real(np.fft.ifft2(np.fft.fft2(a) * np.fft.fft2(field)))


def centred(field, half=6):
    """Roll a field back so its centre is in the middle, then crop for display."""
    n = field.shape[0]
    c = np.roll(field, (n // 2, n // 2), axis=(0, 1))
    m = n // 2
    return c[m - half:m + half + 1, m - half:m + half + 1]


def relu(a):
    return np.maximum(a, 0.0)


def rel_diff(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(a))


# --------------------------------------------------------------------------
# arrays
# --------------------------------------------------------------------------
def make_feature_map(n=N, seed=SEED):
    """A deliberately sharp map: bars, a disc, a checker patch, a step edge."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n]
    h = 0.12 * rng.random((n, n))
    h[:, 4:7] += 0.9                                    # vertical bars
    h[:, 10:12] += 0.7
    h[22:25, :] += 0.6                                  # horizontal bar
    h[(y - 22) ** 2 + (x - 22) ** 2 < 25] += 0.9        # disc
    h[2:10, 20:30] += 0.8 * ((y[2:10, 20:30] + x[2:10, 20:30]) % 2)   # checker
    h += 0.35 * (x > y)                                 # diagonal step
    return h / h.max()


def make_kernel(seed, k=5):
    """A sharp, sign-varying stamp -- visually the opposite of a blur."""
    rng = np.random.default_rng(seed)
    w = rng.uniform(-1.0, 1.0, size=(k, k))
    return w / np.abs(w).sum()


H = make_feature_map()
K = make_kernel(seed=SEED + 1)
K1 = make_kernel(seed=SEED + 2)
K2 = make_kernel(seed=SEED + 3)

Kf, K1f, K2f = as_field(K), as_field(K1), as_field(K2)
Gf = gaussian_field(SIGMA_A)

# Panel A -- the two routes to the same picture
Y = conv(H, Kf)                       # sharp response
Y_blur = conv(Y, Gf)                  # blur applied afterwards
GKf = conv(Kf, Gf)                    # blur folded into the kernel, once
Y_prefold = conv(H, GKf)
RESIDUAL_A = rel_diff(Y_blur, Y_prefold)

# Panel B -- the same move attempted across a ReLU
z = relu(conv(H, K1f))
after = conv(conv(z, K2f), Gf)                    # blur at the output of Conv 2
through_conv2 = conv(z, conv(K2f, Gf))            # blur folded into Conv 2
past_relu = conv(relu(conv(conv(H, K1f), Gf)), K2f)   # blur pushed past the ReLU
RESIDUAL_B_CONV = rel_diff(after, through_conv2)
RESIDUAL_B_RELU = rel_diff(after, past_relu)

# Panel C -- fraction of the map covered by a sigma = 2 px disc
SIZES = (32, 16, 8)
COVER = [np.pi * SIGMA_C ** 2 / (s * s) for s in SIZES]


def pool(a, f):
    n = a.shape[0] // f
    return a.reshape(n, f, n, f).mean(axis=(1, 3))


# --------------------------------------------------------------------------
# layout helpers (figure coordinates)
# --------------------------------------------------------------------------
FIG_W, FIG_H = 11.0, 9.0
ASPECT = FIG_W / FIG_H

ROW1_Y, ROW2_Y = 0.825, 0.590
IMG_W = 0.125
IMG_H = IMG_W * ASPECT
KER_W = 0.085
KER_H = KER_W * ASPECT

X_H, X_K, X_Y, X_OUT = 0.045, 0.235, 0.395, 0.660
X_STAR = X_H + IMG_W + 0.035
X_EQ = X_K + KER_W + 0.038
X_BLUR = 0.590

fig = plt.figure(figsize=(FIG_W, FIG_H))
ov = fig.add_axes([0, 0, 1, 1], facecolor="none", zorder=5)
ov.set_axis_off()
ov.set_xlim(0, 1)
ov.set_ylim(0, 1)


def img_axes(x, y_centre, w, h):
    return fig.add_axes([x, y_centre - h / 2, w, h])


def show(ax, a, cmap, sym=False, frame=None, lw=0.8):
    if sym:
        v = np.abs(a).max()
        ax.imshow(a, cmap=cmap, vmin=-v, vmax=v, interpolation="nearest")
    else:
        ax.imshow(a, cmap=cmap, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(frame or "0.35")
        s.set_linewidth(2.0 if frame else lw)


def glyph(x, y, s, size=20, color="0.2", weight="normal"):
    ov.text(x, y, s, ha="center", va="center", fontsize=size, color=color,
            fontweight=weight)


def arrow(x0, y0, x1, y1, color="0.3", ls="-", lw=1.4, rad=0.0):
    ov.add_patch(FancyArrowPatch((x0, y0), (x1, y1), transform=ov.transAxes,
                                 arrowstyle="-|>", mutation_scale=13,
                                 linewidth=lw, linestyle=ls, color=color,
                                 connectionstyle="arc3,rad=%s" % rad,
                                 shrinkA=0, shrinkB=0))


# --------------------------------------------------------------------------
# Panel A -- the perfect shortcut
# --------------------------------------------------------------------------
show(img_axes(X_H, ROW1_Y, IMG_W, IMG_H), H, "gray")
show(img_axes(X_K, ROW1_Y, KER_W, KER_H), centred(Kf, 4), "RdBu_r", sym=True)
show(img_axes(X_Y, ROW1_Y, IMG_W, IMG_H), Y, "RdBu_r", sym=True)
show(img_axes(X_OUT, ROW1_Y, IMG_W, IMG_H), Y_blur, "RdBu_r", sym=True, frame=BLUE)

show(img_axes(X_H, ROW2_Y, IMG_W, IMG_H), H, "gray")
show(img_axes(X_K, ROW2_Y, KER_W, KER_H), centred(GKf, 4), "RdBu_r", sym=True)
show(img_axes(X_OUT, ROW2_Y, IMG_W, IMG_H), Y_prefold, "RdBu_r", sym=True, frame=BLUE)

for y in (ROW1_Y, ROW2_Y):
    glyph(X_STAR, y, r"$*$", 22)
glyph(X_EQ, ROW1_Y, r"$=$", 22)

# row 1: the blur happens after the convolution
arrow(X_Y + IMG_W + 0.012, ROW1_Y, X_OUT - 0.012, ROW1_Y, lw=1.6)
ov.text((X_Y + IMG_W + X_OUT) / 2, ROW1_Y + 0.030, r"$G_\sigma\,*$",
        ha="center", va="bottom", fontsize=13, color="0.2")

# row 2: a stretched equals sign carries the eye to the same output
for dy in (-0.006, 0.006):
    ov.plot([X_K + KER_W + 0.025, X_OUT - 0.015], [ROW2_Y + dy, ROW2_Y + dy],
            color="0.2", lw=1.8, solid_capstyle="butt")

# the blur migrating from the output onto the kernel
arrow(X_BLUR, ROW1_Y - IMG_H / 2 - 0.012, X_K + KER_W / 2,
      ROW2_Y + KER_H / 2 + 0.012, color=GREY, ls="--", lw=1.3, rad=-0.28)
ov.text(0.435, 0.706, r"$G_\sigma\,*$", ha="center", va="center", fontsize=12,
        color=GREY)

# labels: kernels, and the big equals sign between the two outputs
ov.text(X_K + KER_W / 2, ROW1_Y + KER_H / 2 + 0.012, r"$K$", ha="center",
        va="bottom", fontsize=12, color="0.2")
ov.text(X_K + KER_W / 2, ROW2_Y - KER_H / 2 - 0.014, r"$G_\sigma * K$",
        ha="center", va="top", fontsize=12, color="0.2")
ov.text(X_H + IMG_W / 2, ROW1_Y + IMG_H / 2 + 0.012, r"$H$", ha="center",
        va="bottom", fontsize=12, color="0.2")
ov.text(X_Y + IMG_W / 2, ROW1_Y + IMG_H / 2 + 0.012, r"$K * H$", ha="center",
        va="bottom", fontsize=12, color="0.2")
glyph(X_OUT + IMG_W / 2, (ROW1_Y - IMG_H / 2 + ROW2_Y + IMG_H / 2) / 2,
      r"$=$", 34, BLUE)
ov.text(X_OUT + IMG_W + 0.012, ROW1_Y, r"$G_\sigma*(K*H)$", ha="left",
        va="center", fontsize=11, color=BLUE)
ov.text(X_OUT + IMG_W + 0.012, ROW2_Y, r"$(G_\sigma*K)*H$", ha="left",
        va="center", fontsize=11, color=BLUE)

fig.text(0.02, 0.965, "A", fontsize=17, fontweight="bold", va="center")
fig.text(0.055, 0.965, r"$G_\sigma * (K * H) \;=\; (G_\sigma * K) * H$",
         fontsize=14, va="center")
fig.text(0.98, 0.965, r"$\|\Delta\|_2/\|\cdot\|_2 = %.1e$" % RESIDUAL_A,
         fontsize=10, color=GREY, ha="right", va="center")


# --------------------------------------------------------------------------
# Panel B -- the ReLU roadblock
# --------------------------------------------------------------------------
axB = fig.add_axes([0.045, 0.075, 0.42, 0.33])
axB.set_axis_off()
axB.set_xlim(0, 1)
axB.set_ylim(0, 1)

BOX_Y, BOX_W, BOX_H = 0.70, 0.19, 0.22
centres = [0.20, 0.48, 0.76]
labels = [r"$K_1\,*$", r"$\rho$", r"$K_2\,*$"]
edges = ["0.3", RED, "0.3"]

for cx, lab, ec in zip(centres, labels, edges):
    axB.add_patch(FancyBboxPatch((cx - BOX_W / 2, BOX_Y - BOX_H / 2),
                                 BOX_W, BOX_H,
                                 boxstyle="round,pad=0.012,rounding_size=0.03",
                                 linewidth=1.8, edgecolor=ec,
                                 facecolor="#f4f4f4" if ec == "0.3" else "#fdeaea"))
    axB.text(cx, BOX_Y, lab, ha="center", va="center", fontsize=15)

for x0, x1 in ((0.02, centres[0] - BOX_W / 2 - 0.01),
               (centres[0] + BOX_W / 2 + 0.01, centres[1] - BOX_W / 2 - 0.01),
               (centres[1] + BOX_W / 2 + 0.01, centres[2] - BOX_W / 2 - 0.01),
               (centres[2] + BOX_W / 2 + 0.01, 0.86)):
    axB.add_patch(FancyArrowPatch((x0, BOX_Y), (x1, BOX_Y), arrowstyle="-|>",
                                  mutation_scale=12, linewidth=1.3,
                                  color="0.3", shrinkA=0, shrinkB=0))

axB.text(0.02, BOX_Y + 0.10, r"$H$", ha="left", va="center", fontsize=12)
axB.add_patch(Circle((0.93, BOX_Y), 0.055, facecolor="#e8f0f8",
                     edgecolor=BLUE, linewidth=1.8))
axB.text(0.93, BOX_Y, r"$G_\sigma$", ha="center", va="center", fontsize=12,
         color=BLUE)

# the blur trying to slide backwards
SLIDE_Y = 0.36
axB.add_patch(FancyArrowPatch((0.93, SLIDE_Y), (centres[2], SLIDE_Y),
                              arrowstyle="-|>", mutation_scale=12,
                              linewidth=1.5, linestyle="--", color=BLUE,
                              shrinkA=0, shrinkB=0))
axB.add_patch(FancyArrowPatch((centres[2], SLIDE_Y), (centres[1] + 0.055, SLIDE_Y),
                              arrowstyle="-|>", mutation_scale=12,
                              linewidth=1.5, linestyle="--", color=BLUE,
                              shrinkA=0, shrinkB=0))
for cx, col in ((centres[2], BLUE), (0.93, BLUE)):
    axB.plot([cx, cx], [SLIDE_Y + 0.03, BOX_Y - BOX_H / 2 - 0.02], ls=":",
             lw=1.0, color=col)
axB.plot([centres[1], centres[1]], [SLIDE_Y + 0.06, BOX_Y - BOX_H / 2 - 0.02],
         ls=":", lw=1.0, color=RED)

# the roadblock itself
axB.plot([centres[1] - 0.045, centres[1] + 0.045],
         [SLIDE_Y - 0.045, SLIDE_Y + 0.045], color=RED, lw=4.5,
         solid_capstyle="round")
axB.plot([centres[1] - 0.045, centres[1] + 0.045],
         [SLIDE_Y + 0.045, SLIDE_Y - 0.045], color=RED, lw=4.5,
         solid_capstyle="round")
axB.text(centres[2], SLIDE_Y - 0.10, u"✓", ha="center", va="center",
         fontsize=15, color=GREEN)

axB.text(0.80, SLIDE_Y - 0.24,
         r"$G_\sigma*(K_2*z)=(G_\sigma*K_2)*z$", ha="center", va="center",
         fontsize=9.5, color=GREEN)
axB.text(0.80, SLIDE_Y - 0.34, r"$%.0e$" % RESIDUAL_B_CONV,
         ha="center", va="center", fontsize=9, color=GREY)
axB.text(0.36, SLIDE_Y - 0.24,
         r"$G_\sigma*\rho(v)\;\neq\;\rho(G_\sigma*v)$", ha="center",
         va="center", fontsize=9.5, color=RED)
axB.text(0.36, SLIDE_Y - 0.34, r"$%.0f\%%$" % (100 * RESIDUAL_B_RELU),
         ha="center", va="center", fontsize=9, color=RED)

fig.text(0.02, 0.445, "B", fontsize=17, fontweight="bold", va="center")
fig.text(0.055, 0.445, r"$G_\sigma \circ \rho \;\neq\; \rho \circ G_\sigma$",
         fontsize=14, va="center")


# --------------------------------------------------------------------------
# Panel C -- the shrinking canvas (constant cell pitch)
# --------------------------------------------------------------------------
axC = fig.add_axes([0.545, 0.085, 0.44, 0.295])
axC.set_axis_off()
axC.set_aspect("equal")

gap = 4.0
x0 = 0.0
for i, (s, cov) in enumerate(zip(SIZES, COVER)):
    a = pool(H, N // s)
    axC.imshow(a, cmap="gray", interpolation="nearest",
               extent=(x0, x0 + s, 0, s), zorder=0)
    for t in range(s + 1):
        axC.plot([x0, x0 + s], [t, t], color="w", lw=0.25, alpha=0.45, zorder=1)
        axC.plot([x0 + t, x0 + t], [0, s], color="w", lw=0.25, alpha=0.45, zorder=1)
    axC.add_patch(Rectangle((x0, 0), s, s, fill=False, edgecolor="0.3",
                            lw=1.0, zorder=2))
    axC.add_patch(Circle((x0 + s / 2, s / 2), SIGMA_C, facecolor=RED,
                         alpha=0.30, edgecolor=RED, lw=1.8, zorder=3))
    axC.text(x0 + s / 2, s + 1.0, r"$%d\times%d$" % (s, s), ha="center",
             va="bottom", fontsize=11)
    axC.text(x0 + s / 2, -1.2, r"$%.1f\%%$" % (100 * cov), ha="center",
             va="top", fontsize=11, color=RED)
    x0 += s + gap

axC.text(SIZES[0] / 2, SIZES[0] / 2 - SIGMA_C - 1.4,
         r"$\sigma = %g$ px" % SIGMA_C, ha="center", va="top", fontsize=11,
         color=RED, bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                              edgecolor="none", alpha=0.85))
axC.set_xlim(-1.5, x0 - gap + 1.5)
axC.set_ylim(-7.0, SIZES[0] + 5.0)

fig.text(0.52, 0.445, "C", fontsize=17, fontweight="bold", va="center")
fig.text(0.555, 0.445, r"$\sigma$ fixed in layer pixels $\;\Rightarrow\;$ "
                       r"$\pi\sigma^2/(HW)$ grows $\times%d$"
                       % round(COVER[-1] / COVER[0]), fontsize=14, va="center")


# --------------------------------------------------------------------------
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / ("%s.%s" % (NAME, ext)), dpi=200)
    plt.close(fig)
    print("panel A residual   ||G*(K*H) - (G*K)*H|| / ||.|| = %.3e" % RESIDUAL_A)
    print("panel B through conv2                            = %.3e" % RESIDUAL_B_CONV)
    print("panel B across ReLU                              = %.1f%%"
          % (100 * RESIDUAL_B_RELU))
    print("panel C coverage   " + "  ".join("%dx%d: %.1f%%" % (s, s, 100 * c)
                                            for s, c in zip(SIZES, COVER)))
    for ext in ("png", "pdf", "svg"):
        print("wrote", OUT / ("%s.%s" % (NAME, ext)))


if __name__ == "__main__":
    main()
