"""Visual verification of the input transformation."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from .data import fixed_subset_indices  # noqa: E402
from .pipeline import InputPipeline  # noqa: E402


def class_diverse_indices(labels, per_class: int = 1, seed: int = 777) -> np.ndarray:
    """One (or more) fixed example per class, reproducibly chosen."""
    labels = np.asarray(labels)
    n_classes = int(labels.max()) + 1
    return fixed_subset_indices(labels.shape[0], per_class * n_classes, seed,
                                "viz_examples", labels=labels)


def level_grid(images_uint8: torch.Tensor, labels, class_names, transform,
               levels, out_path, title: str | None = None) -> Path:
    """Rows = images, columns = transformation levels.  Images stay in float space."""
    x = InputPipeline.to_unit_float(images_uint8)
    rows, cols = x.shape[0], len(levels)
    fig, axes = plt.subplots(rows, cols, figsize=(1.35 * cols, 1.45 * rows), squeeze=False)
    for j, eta in enumerate(levels):
        y = transform(x, eta).clamp(0, 1).cpu()
        for i in range(rows):
            ax = axes[i][j]
            ax.imshow(y[i].permute(1, 2, 0).numpy(), interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                label = "%s=%g" % (transform.parameter_name, eta)
                if transform.is_target(eta):
                    label += "\n(target, identity)"
                ax.set_title(label, fontsize=9)
            if j == 0:
                name = class_names[int(labels[i])] if class_names is not None else str(int(labels[i]))
                ax.set_ylabel(name, fontsize=8, rotation=0, ha="right", va="center")
    fig.suptitle(title or "Gaussian input smoothing, fixed class-diverse training images",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def boundary_effect_figure(transform, levels, out_path, size: int = 32) -> Path:
    """Show what reflection padding does at the borders, on a constant image and a step.

    The constant image must come back *exactly* constant; the step image shows
    the mirror-symmetry bias within ``radius`` pixels of the edge.
    """
    const = torch.full((1, 1, size, size), 0.5)
    step = torch.zeros((1, 1, size, size))
    step[..., : size // 2] = 1.0
    fig, axes = plt.subplots(2, len(levels) + 1, figsize=(1.5 * (len(levels) + 1), 3.4),
                             squeeze=False)
    for row, (img, name) in enumerate([(const, "constant 0.5"), (step, "vertical step")]):
        axes[row][0].plot(img[0, 0, size // 2].numpy(), lw=1.2)
        axes[row][0].set_title("%s\ncentre row" % name, fontsize=8)
        axes[row][0].set_ylim(-0.05, 1.05)
        for j, eta in enumerate(levels):
            y = transform(img, eta)
            ax = axes[row][j + 1]
            ax.plot(y[0, 0, size // 2].numpy(), lw=1.2)
            ax.set_ylim(-0.05, 1.05)
            ax.set_title("%s=%g" % (transform.parameter_name, eta), fontsize=8)
            ax.set_xticks([0, size - 1])
    fig.suptitle("Reflection-padding boundary behaviour (centre-row profiles)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def attenuation_figure(transform, levels, out_path, size: int = 64) -> Path:
    """Measured attenuation of sinusoidal gratings vs the continuous prediction.

    The dashed curves are ``exp(-sigma^2 w^2 / 2)`` for the *continuous* Gaussian.
    The measured points come from the finite, truncated, reflection-padded
    discrete filter and are expected to deviate, especially at high frequency;
    the figure documents that deviation rather than asserting the two agree.
    """
    freqs = np.arange(1, size // 2)
    xs = np.arange(size)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for c, eta in enumerate([e for e in levels if e > 0]):
        measured = []
        for k in freqs:
            grating = 0.5 + 0.4 * np.cos(2 * np.pi * k * xs / size)
            img = torch.tensor(np.tile(grating, (size, 1)), dtype=torch.float32)[None, None]
            y = transform(img, eta)
            inner = y[0, 0, size // 2, size // 4: 3 * size // 4].numpy()
            amp = (inner.max() - inner.min()) / 2.0
            measured.append(amp / 0.4)
        w = 2 * np.pi * freqs / size
        ax.plot(w, measured, "o-", ms=3, lw=1.2, color=colors[c % len(colors)],
                label="measured, sigma=%g" % eta)
        ax.plot(w, np.exp(-(eta ** 2) * w ** 2 / 2), "--", lw=1.0,
                color=colors[c % len(colors)], label="continuous exp(-s^2 w^2/2), sigma=%g" % eta)
    ax.set_xlabel("angular frequency w (rad/pixel)")
    ax.set_ylabel("amplitude retained")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 2)
    ax.set_title("Attenuation of spatial oscillations (interior of the image)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
