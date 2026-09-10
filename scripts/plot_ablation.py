"""Figures for the anti-aliasing ablation.

Two PNGs under ``results/``:

* ``ablation_aa_curves.png``     -- test-accuracy trajectories, one panel per test
* ``ablation_aa_synthesis.png``  -- sigma sweep, CE, and the two diagnostics

Plotting rules taken from ``docs/HANDOVER.md`` section 4, which records what went
wrong the last time these curves were drawn:

* the **current path** (the configuration the weights and BN buffers were actually
  trained under) is the primary, solid curve; the **target path** is a thin dashed
  diagnostic.  While a filter is active the target path measures a premature
  configuration change including BatchNorm mismatch -- at epoch 6 of the plateau
  arm it reads CE 9.42 against 1.08 on the current path -- so **axes are clipped to
  the primary curves** or every real difference flattens to nothing;
* every line style appears in the legend;
* the rendered PNG is read back rather than trusted from the script.

Colour follows the validated categorical palette in the ``dataviz`` reference
(slots 1-6, light mode); the two controls keep fixed slots in every panel so an
entity never changes colour between views.  Curves carry direct end labels, which
is the relief the palette's contrast warning requires.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUTPUTS = ROOT / "results" / "kaggle_outputs"

# validated categorical palette, light mode (dataviz references/palette.md)
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a887f"
SURFACE = "#fcfcfb"

# controls keep fixed slots in every panel
COLOR = {"C_plain": SLOT[0], "C_plateau": SLOT[1]}

PANELS = [
    ("Tests 3+4 - sigma constant contre sigma annele",
     ["C_plain", "C_plateau", "K_const030", "K_const050", "K_const080", "K_const100"],
     "un sigma constant n'atteint jamais l'endpoint cible : ce n'est pas une continuation"),
    ("Test 1 - placement du filtre",
     ["C_plain", "C_plateau", "P_postbn", "P_postblock"],
     "post_block = apres le ReLU, la position correcte pour un anti-aliasing (10 positions, pas 19)"),
    ("Test 2 - quels sites",
     ["C_plain", "C_plateau", "M_predown", "M_nodown"],
     "predown = sites {6,12} seuls ; nodown = les 17 autres"),
    ("Test 5 - vrai prefiltre anti-aliasing fixe",
     ["C_plain", "C_plateau", "B_blurpool", "B_blurpool_plateau"],
     "BlurPool : sigma 0.5 fixe sur les entrees de blocks[3] et blocks[6]"),
]

LABEL = {"C_plain": "plain", "C_plateau": "plateau 19 sites",
         "P_postbn": "post-BN", "P_postblock": "post-ReLU",
         "M_predown": "predown {6,12}", "M_nodown": "nodown (17)",
         "K_const030": "const 0.30", "K_const050": "const 0.50",
         "K_const080": "const 0.80", "K_const100": "const 1.00",
         "B_blurpool": "blurpool seul", "B_blurpool_plateau": "blurpool+plateau"}


def load():
    arms = {}
    for f in sorted(OUTPUTS.glob("abl-j*/**/metrics.json")):
        cid = f.parent.name.split("__seed")[0]
        arms[cid] = json.loads(f.read_text())
    res = json.loads((ROOT / "results" / "ablation_aa_results.json").read_text())
    return arms, res["arms"]


def color_for(cid, panel_ids):
    if cid in COLOR:
        return COLOR[cid]
    rest = [c for c in panel_ids if c not in COLOR]
    return SLOT[2 + rest.index(cid)]


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color="#e6e5e0", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d5d4ce")
    ax.tick_params(colors=INK2, labelsize=8, length=3)


def declutter(items, min_gap):
    """Nudge (y, label, colour) end labels apart, preserving their order."""
    items = sorted(items, key=lambda t: t[0])
    ys = [t[0] for t in items]
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < min_gap:
            ys[i] = ys[i - 1] + min_gap
    return [(ys[i], items[i][1], items[i][2]) for i in range(len(items))]


def scatter_labels(ax, pts, min_gap, dx=0.008):
    """Label a scatter without overlaps: nudge in y, draw a leader when moved."""
    pts = sorted(pts, key=lambda t: t[1])
    ys = [t[1] for t in pts]
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < min_gap:
            ys[i] = ys[i - 1] + min_gap
    span = ax.get_xlim()[1] - ax.get_xlim()[0]
    for (x, y, lab, c), ynew in zip(pts, ys):
        ax.annotate(lab, (x + dx * span, ynew), color=INK, fontsize=7.5,
                    va="center", ha="left", zorder=5)
        if abs(ynew - y) > 1e-12:
            ax.plot([x, x + dx * span * 0.85], [y, ynew], color=c, lw=0.7,
                    alpha=0.6, zorder=2)
    # the nudge can push a label past the current top; widen rather than clip it
    lo0, hi0 = ax.get_ylim()
    pad = (hi0 - lo0) * 0.04
    ax.set_ylim(min(lo0, min(ys) - pad), max(hi0, max(ys) + pad))


def curves_figure(arms, summary, dest):
    """Primary (current-path) trajectories only, one panel per test.

    The target path is deliberately absent here: in every panel its message is
    "do not read this", and drawing it four times compressed the informative
    band into the top fifth of each axis.  It gets one dedicated demonstration
    in the synthesis figure instead.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.0), facecolor=SURFACE)
    for ax, (title, ids, sub) in zip(axes.ravel(), PANELS):
        style_axis(ax)
        lo, hi, ends = 1.0, 0.0, []
        for cid in ids:
            m = arms.get(cid)
            if not m:
                continue
            c = color_for(cid, ids)
            e = [r["epoch"] for r in m]
            cur = [r["test_acc_current"] for r in m]
            # epoch 0 is the untrained network (~0.10) and would compress the
            # whole informative band; the axis starts after it
            lo = min(lo, min(cur[1:])); hi = max(hi, max(cur))
            ax.plot(e, cur, color=c, linewidth=2.0, zorder=3,
                    solid_capstyle="round")
            ends.append((cur[-1], LABEL[cid], c))
        span = hi - lo
        # declutter first, THEN bound the axis: nudged labels can sit outside the
        # curve range, and a label pushed past the top is silently lost
        placed = declutter(ends, span * 0.045)
        ylo = min([lo] + [y for y, _, _ in placed]) - span * 0.06
        yhi = max([hi] + [y for y, _, _ in placed]) + span * 0.06
        for y, lab, c in placed:
            ax.annotate(" " + lab, (30.4, y), color=c, fontsize=8.5,
                        va="center", ha="left", fontweight="bold", zorder=4)
        ax.axvline(21, color=INK3, linewidth=0.8, linestyle=":", zorder=1)
        ax.annotate("extinction e21", (20.5, ylo + span * 0.02), color=INK3,
                    fontsize=7, rotation=90, va="bottom", ha="right")
        ax.set_ylim(ylo, yhi)
        ax.set_xlim(1.4, 38)
        ax.set_title(title, color=INK, fontsize=11.5, fontweight="bold",
                     loc="left", pad=24)
        ax.annotate(sub, xy=(0, 1.018), xycoords="axes fraction",
                    color=INK2, fontsize=8.5, va="bottom")
        ax.set_xlabel("epoque", color=INK2, fontsize=9)
        ax.set_ylabel("accuracy test (chemin courant)", color=INK2, fontsize=9)

    fig.suptitle("Ablation anti-aliasing - ResNet-20 BN, CIFAR-10 50k/10k, 30 epoques, 1 graine",
                 color=INK, fontsize=13.5, fontweight="bold", y=0.988)
    fig.text(0.5, 0.958,
             "courbes = chemin courant, la configuration reellement entrainee (BN comprise). "
             "Axe borne apres l'epoque 0. Une seule graine : signes et ordres de grandeur, aucune significativite.",
             color=INK2, fontsize=9, ha="center")
    fig.tight_layout(rect=[0, 0.005, 1, 0.945])
    fig.savefig(dest, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def synthesis_figure(arms, summary, dest):
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 10.5), facecolor=SURFACE)
    base = summary["C_plain"]["acc"]
    plat = summary["C_plateau"]["acc"]

    def head(ax, title, sub):
        ax.set_title(title, color=INK, fontsize=11.5, fontweight="bold",
                     loc="left", pad=24)
        ax.annotate(sub, xy=(0, 1.018), xycoords="axes fraction", color=INK2,
                    fontsize=8.5, va="bottom")

    # --- E : balayage de sigma constant ------------------------------------
    ax = axes[0, 0]; style_axis(ax)
    sig = [0.30, 0.50, 0.80, 1.00]
    ids = ["K_const030", "K_const050", "K_const080", "K_const100"]
    acc = [summary[i]["acc"] for i in ids]
    ax.axhline(base, color=COLOR["C_plain"], lw=1.4, ls="--", zorder=2)
    ax.annotate("plain", (0.245, base), color=COLOR["C_plain"], fontsize=8.5,
                fontweight="bold", va="bottom")
    ax.axhline(plat, color=COLOR["C_plateau"], lw=1.4, ls=":", zorder=2)
    ax.annotate("plateau annele", (0.245, plat), color=COLOR["C_plateau"],
                fontsize=8.5, fontweight="bold", va="bottom")
    ax.plot(sig, acc, color=SLOT[2], lw=2.0, marker="o", markersize=9,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
    for s_, a in zip(sig, acc):
        ax.annotate("%.4f" % a, (s_, a), textcoords="offset points",
                    xytext=(0, 12), ha="center", color=INK, fontsize=8.5,
                    fontweight="bold")
    ax.set_xlim(0.23, 1.10)
    ax.set_ylim(min(acc) - 0.014, max(acc + [plat]) + 0.017)
    head(ax, "Optimum de sigma a filtre constant",
         "un pic intermediaire est la signature attendue d'un anti-aliasing (Nyquist)")
    ax.set_xlabel("sigma constant (pixels)", color=INK2, fontsize=9)
    ax.set_ylabel("accuracy test finale", color=INK2, fontsize=9)

    # --- F : CE, chemin courant --------------------------------------------
    ax = axes[0, 1]; style_axis(ax)
    keys = ["C_plain", "C_plateau", "P_postblock", "B_blurpool", "K_const050"]
    lo, hi, ends = 9e9, 0, []
    for cid in keys:
        m = arms[cid]; c = color_for(cid, keys)
        e = [r["epoch"] for r in m]; ce = [r["test_ce_current"] for r in m]
        lo, hi = min(lo, min(ce)), max(hi, max(ce[1:]))
        ax.plot(e, ce, color=c, lw=2.0, zorder=3)
        ends.append((ce[-1], LABEL[cid], c))
    placed = declutter(ends, (hi - lo) * 0.05)
    for y, lab, c in placed:
        ax.annotate(" " + lab, (30.4, y), color=c, fontsize=8.5, va="center",
                    fontweight="bold")
    ax.set_ylim(min([lo * 0.90] + [y for y, _, _ in placed]),
                max([hi * 1.04] + [y for y, _, _ in placed]) * 1.02)
    ax.set_xlim(1.4, 41)
    head(ax, "Cross-entropy test, chemin courant",
         "axe borne aux courbes primaires")
    ax.set_xlabel("epoque", color=INK2, fontsize=9)
    ax.set_ylabel("CE test", color=INK2, fontsize=9)

    # --- G : le piege du chemin cible --------------------------------------
    ax = axes[0, 2]; style_axis(ax)
    for cid, c in (("C_plateau", COLOR["C_plateau"]), ("K_const080", SLOT[5])):
        m = arms[cid]
        e = [r["epoch"] for r in m]
        ax.plot(e, [r["test_acc_current"] for r in m], color=c, lw=2.0, zorder=3)
        ax.plot(e, [r["test_acc_target"] for r in m], color=c, lw=1.1,
                ls=(0, (3, 3)), alpha=0.75, zorder=2)
        ax.annotate(" " + LABEL[cid], (30.4, m[-1]["test_acc_current"]),
                    color=c, fontsize=8.5, va="center", fontweight="bold")
    ax.annotate("0.10 = hasard", (3.0, arms["K_const080"][-1]["test_acc_target"]),
                textcoords="offset points", xytext=(0, 7), color=SLOT[5],
                fontsize=8, va="bottom", ha="left", alpha=0.9)
    ax.set_xlim(1.4, 41); ax.set_ylim(0.05, 0.85)
    ax.legend(handles=[
        plt.Line2D([], [], color=INK2, lw=2.0, label="chemin courant (primaire)"),
        plt.Line2D([], [], color=INK2, lw=1.1, ls=(0, (3, 3)), alpha=0.75,
                   label="chemin cible (diagnostic)")],
        frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
    head(ax, "Pourquoi le chemin cible ne doit pas etre lu",
         "filtre arrache alors que BatchNorm a accumule avec : le bras constant tombe au hasard")
    ax.set_xlabel("epoque", color=INK2, fontsize=9)
    ax.set_ylabel("accuracy test", color=INK2, fontsize=9)

    # --- H : consistance au decalage ---------------------------------------
    ax = axes[1, 0]; style_axis(ax)
    pts = []
    for cid in LABEL:
        a = summary[cid]
        c = COLOR.get(cid, SLOT[2] if a["acc"] >= base else SLOT[4])
        ax.plot(a["shift_consistency"], a["acc"], marker="o", markersize=9,
                color=c, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        pts.append((a["shift_consistency"], a["acc"], LABEL[cid], c))
    ax.axhline(base, color=COLOR["C_plain"], lw=1.0, ls="--", zorder=1)
    ax.set_xlim(0.79, 0.965)
    accs = [t[1] for t in pts]
    ax.set_ylim(min(accs) - 0.006, max(accs) + 0.006)
    scatter_labels(ax, pts, (max(accs) - min(accs)) * 0.055)
    head(ax, "Diagnostic 0b - consistance au decalage de 1 px",
         "l'anti-aliasing predit plus de stabilite ; en acheter trop ne paie pas")
    ax.set_xlabel("consistance top-1 sous translation", color=INK2, fontsize=9)
    ax.set_ylabel("accuracy test finale", color=INK2, fontsize=9)

    # --- I : energie repliee -----------------------------------------------
    ax = axes[1, 1]; style_axis(ax)
    ids = [c for c in LABEL if summary[c]["alias"].get("blocks3") is not None]
    x = list(range(len(ids))); w = 0.38
    ax.bar([i - w / 2 - 0.012 for i in x], [summary[c]["alias"]["blocks3"] for c in ids],
           w, color=SLOT[0], zorder=3, label="entree blocks[3]  (32->16)")
    ax.bar([i + w / 2 + 0.012 for i in x], [summary[c]["alias"]["blocks6"] for c in ids],
           w, color=SLOT[3], zorder=3, label="entree blocks[6]  (16->8)")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[c] for c in ids], rotation=40, ha="right", fontsize=7.5)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right")
    head(ax, "Diagnostic 0a - energie au-dessus de la Nyquist",
         "fraction de |DFT|^2 avec |f|>0.25 c/ech : ce qui se replie a la decimation")
    ax.set_ylabel("fraction d'energie repliee", color=INK2, fontsize=9)

    # --- J : cout contre gain ----------------------------------------------
    ax = axes[1, 2]; style_axis(ax)
    wp = summary["C_plain"]["wall_s"]
    cg = []
    for cid in LABEL:
        if cid == "C_plain":
            continue
        a = summary[cid]
        c = COLOR.get(cid, SLOT[2] if a["acc"] >= base else SLOT[4])
        ax.plot(a["wall_s"] / wp, (a["acc"] - base) * 100, marker="o",
                markersize=9, color=c, markeredgecolor=SURFACE,
                markeredgewidth=2, zorder=3)
        cg.append((a["wall_s"] / wp, (a["acc"] - base) * 100, LABEL[cid], c))
    ax.axhline(0, color=COLOR["C_plain"], lw=1.0, ls="--", zorder=1)
    gains = [t[1] for t in cg]
    ax.set_ylim(min(gains) - 0.5, max(gains) + 0.7)
    scatter_labels(ax, cg, (max(gains) - min(gains)) * 0.052)
    ax.annotate("plain", (1.0, 0), textcoords="offset points", xytext=(4, 5),
                color=COLOR["C_plain"], fontsize=8, fontweight="bold")
    ax.set_xlim(0.95, 2.0)
    head(ax, "Cout contre gain",
         "temps total rapporte au plain ; le prefiltre fixe est le seul gain quasi gratuit")
    ax.set_xlabel("temps du run / temps du plain", color=INK2, fontsize=9)
    ax.set_ylabel("gain d'accuracy (points)", color=INK2, fontsize=9)

    fig.suptitle("Ablation anti-aliasing - synthese et diagnostics", color=INK,
                 fontsize=13.5, fontweight="bold", y=0.988)
    fig.tight_layout(rect=[0, 0.005, 1, 0.955])
    fig.savefig(dest, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    arms, summary = load()
    d1 = ROOT / "results" / "ablation_aa_curves.png"
    d2 = ROOT / "results" / "ablation_aa_synthesis.png"
    curves_figure(arms, summary, d1)
    synthesis_figure(arms, summary, d2)
    print("ecrit ->", d1)
    print("ecrit ->", d2)


if __name__ == "__main__":
    main()
