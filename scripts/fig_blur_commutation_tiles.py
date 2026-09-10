"""Render the raster tiles and the measured numbers for the TikZ figure
``figures/blur_commutation/blur_commutation.tex``.

The figure explains where the Gaussian-blur shortcut

    G_s * (K * H) = (G_s * K) * H

holds and where it stops holding inside a network.  This script produces only
the pixel content: a real CIFAR-10 test image, the convolution stamps, the
responses, the two disagreeing branches around a ReLU, and the three pooled
maps of the striding panel.  All layout, type and arrows live in the .tex.

Every convolution is an exact FFT circular convolution, so the residuals
printed here (and written into ``numbers.tex``) are measurements, not claims.
Response tiles are each shown at their own symmetric colour scale -- the two
outputs of the identity are bit-identical arrays, so they render identically.

Tiles are written at an integer upsampling factor with nearest-neighbour
replication, so the PNGs carry hard pixel edges and no PDF viewer smooths the
CIFAR pixels away.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import numpy as np
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parents[1]
CIFAR = ROOT / "data" / "cifar-10-batches-py" / "test_batch"
OUT = ROOT / "figures" / "blur_commutation" / "tiles"

IMAGE_INDEX = 9520      # a clean, high-contrast automobile: legible at 8x8
N = 32
SIGMA = 1.5             # blur width for panels A and B, in map pixels
SIGMA_C = 2.0           # blur width advertised in the striding panel
TAU = 1.2               # scale of the oriented edge stamp K
THETA = np.deg2rad(35)  # its orientation
KHALF = 6               # stamps are cropped to (2*KHALF+1)^2
UPSAMPLE = 14           # nearest-neighbour replication factor for the PNGs
GRID_RGB = (0.78, 0.81, 0.84)   # baked pixel-grid colour for panel C

SIZES = (32, 16, 8)


# --------------------------------------------------------------------------
# exact circular convolution on the N x N torus; kernels are image-sized
# fields whose centre sits at index (0, 0).
# --------------------------------------------------------------------------
def gaussian_field(sigma, n=N):
    d = np.minimum(np.arange(n), n - np.arange(n))
    g1 = np.exp(-0.5 * (d / sigma) ** 2)
    g = np.outer(g1, g1)
    return g / g.sum()


def edge_field(tau, theta, n=N):
    """Derivative of a Gaussian along theta: a sharp, sign-varying stamp."""
    d = np.minimum(np.arange(n), n - np.arange(n)).astype(float)
    sy = np.where(np.arange(n) <= n // 2, 1.0, -1.0)
    yy = (d * sy)[:, None]
    xx = (d * sy)[None, :]
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * tau ** 2))
    k = -(xx * np.cos(theta) + yy * np.sin(theta)) / tau ** 2 * g
    return k / np.abs(k).sum()


def conv(a, field):
    return np.real(np.fft.ifft2(np.fft.fft2(a) * np.fft.fft2(field)))


def crop(field, half=KHALF):
    n = field.shape[0]
    c = np.roll(field, (n // 2, n // 2), axis=(0, 1))
    m = n // 2
    return c[m - half:m + half + 1, m - half:m + half + 1]


def relu(a):
    return np.maximum(a, 0.0)


def rel_diff(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(a))


def pool(a, f):
    n = a.shape[0] // f
    return a.reshape(n, f, n, f).mean(axis=(1, 3))


# --------------------------------------------------------------------------
# tile writing
# --------------------------------------------------------------------------
def rgb_gray(a):
    v = Normalize(vmin=a.min(), vmax=a.max())(a)
    return np.repeat(v[:, :, None], 3, axis=2)


def rgb_signed(a, cmap="RdBu_r"):
    v = np.abs(a).max()
    return matplotlib.colormaps[cmap](Normalize(-v, v)(a))[:, :, :3]


def write(name, rgb, grid=False, factor=UPSAMPLE):
    big = np.repeat(np.repeat(rgb, factor, axis=0), factor, axis=1)
    if grid:
        big[::factor, :, :] = GRID_RGB
        big[:, ::factor, :] = GRID_RGB
    img = (np.clip(big, 0, 1) * 255).astype(np.uint8)
    # raw pixels straight to PNG: no figure, no axes, no padding
    mpimg.imsave(OUT / (name + ".png"), img)
    return name


# --------------------------------------------------------------------------
def main():
    OUT.mkdir(parents=True, exist_ok=True)

    d = pickle.load(open(CIFAR, "rb"), encoding="bytes")
    rgb = d[b"data"][IMAGE_INDEX].reshape(3, N, N).astype(np.float64) / 255.0
    H = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]

    Kf = edge_field(TAU, THETA)
    Gf = gaussian_field(SIGMA)
    GKf = conv(Kf, Gf)

    # panel A: the two routes
    KH = conv(H, Kf)
    KH_blur = conv(KH, Gf)
    out = conv(H, GKf)
    res_a = rel_diff(KH_blur, out)

    # panel B: the same move across a ReLU
    K1f = edge_field(TAU, THETA)
    K2f = edge_field(TAU, THETA + np.pi / 2)
    z = relu(conv(H, K1f))
    after = conv(conv(z, K2f), Gf)                        # blur at the output
    through = conv(z, conv(K2f, Gf))                      # blur folded into K2
    past = conv(relu(conv(conv(H, K1f), Gf)), K2f)        # blur pushed past rho
    res_b_conv = rel_diff(after, through)
    res_b_relu = rel_diff(after, past)

    write("H", rgb_gray(H))
    write("K", rgb_signed(crop(Kf)))
    write("GK", rgb_signed(crop(GKf)))
    write("KH", rgb_signed(KH))
    write("KHblur", rgb_signed(KH_blur))
    write("out", rgb_signed(out))
    write("relu_after", rgb_signed(after))
    write("relu_past", rgb_signed(past))
    write("relu_diff", rgb_signed(after - past))

    cover = []
    for s in SIZES:
        m = pool(H, N // s)
        write("c%d" % s, rgb_gray(m), grid=True, factor=UPSAMPLE * (N // s))
        cover.append(np.pi * SIGMA_C ** 2 / (s * s))

    def sci(x):
        """Render a float as TeX scientific notation, e.g. 2.8\\times 10^{-16}."""
        m, e = ("%.1e" % x).split("e")
        return "%s\\times 10^{%d}" % (m, int(e))

    numbers = OUT.parent / "numbers.tex"
    numbers.write_text(
        "%% generated by scripts/fig_blur_commutation_tiles.py -- do not edit\n"
        "\\newcommand{\\resIdentity}{%s}\n"
        "\\newcommand{\\resConvSlide}{%s}\n"
        "\\newcommand{\\resReluBreak}{%.0f}\n"
        "\\newcommand{\\coverThirtyTwo}{%.1f}\n"
        "\\newcommand{\\coverSixteen}{%.1f}\n"
        "\\newcommand{\\coverEight}{%.1f}\n"
        "\\newcommand{\\coverRatio}{%d}\n"
        "\\newcommand{\\blurSigma}{%g}\n"
        "\\newcommand{\\blurSigmaC}{%g}\n"
        "\\newcommand{\\cifarIndex}{%d}\n"
        % (sci(res_a), sci(res_b_conv), 100 * res_b_relu, 100 * cover[0],
           100 * cover[1], 100 * cover[2], round(cover[2] / cover[0]),
           SIGMA, SIGMA_C, IMAGE_INDEX))

    print("identity residual        %.3e" % res_a)
    print("blur through conv2       %.3e" % res_b_conv)
    print("blur past ReLU           %.1f%%" % (100 * res_b_relu))
    print("coverage 32/16/8         " + " ".join("%.1f%%" % (100 * c) for c in cover))
    print("tiles ->", OUT)
    print("numbers ->", numbers)


if __name__ == "__main__":
    main()
