"""Generate panel A of the progressive-resolution figure with PlotNeuralNet pics.

Writes ``figures/resolution/panelA.tex``: the two reduction paths drawn as two
rows of the same network, with box height tracking the real spatial extent, so
the shrink is visible rather than asserted.

    input_*  shrinks the float image before normalization, so every site --
             the stem included -- sees r x r.
    stem_*   keeps the input at 32, and shrinks the stem's output instead, so
             the stem still sees 32 x 32 and only the blocks are reduced.

Resolutions come from ``scripts.campaign_manifest.RESOLUTIONS``; stage widths
follow the network's own stride pattern.  The fragment is included inside the
tikzpicture of ``figures/resolution/body.tex``.

Box/Ball pics: PlotNeuralNet (MIT), vendored under
``figures/gaussian_sites_pnn/layers/``.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from continuation.campaign_ops import REDUCTIONS  # noqa: E402
from scripts.campaign_manifest import RESOLUTIONS  # noqa: E402

OUT = ROOT / "figures" / "resolution" / "panelA.tex"

FULL = 32                      # the unreduced input side
SHOWN_R = 16                   # the reduction illustrated (the low leg of Rprog)
N_STAGES = 3
HSCALE = 0.42                  # drawn box height per pixel of feature-map side
DEPTH_F = 0.42
PITCH = 0.95
GATE_PITCH = 1.7               # clearance around the growing gate
ROW_DY = 5.8
ROW_X = 1.9                    # left margin for the row, clear of captions


def box(name, to, offset, h, w, fill, depth_f=DEPTH_F, opacity=None):
    return ("\\pic[shift={%s}] at %s\n"
            "    {Box={name=%s, caption= , xlabel={{, }}, zlabel= ,\n"
            "          fill=%s, %sheight=%g, width=%g, depth=%g}};\n"
            % (offset, to, name, fill,
               "" if opacity is None else "opacity=%g, " % opacity,
               h, w, h * depth_f))


def schedule_runs(name="Rprog"):
    """The (start epoch, end epoch, resolution) plateaus of a schedule."""
    v = RESOLUTIONS[name]
    runs, st = [], 0
    for e in range(1, len(v) + 1):
        if e == len(v) or v[e] != v[st]:
            runs.append((st, e, v[st]))
            st = e
    return runs


def growing_gate(prefix, anchor, dx, runs):
    """The reduction gate drawn as nested rings, one per plateau of r(e).

    This is where the continuation actually lives: the layer is not a fixed
    resize, it grows 16 -> 24 -> 32 as training proceeds, and at the last
    plateau it is the identity.  Largest ring first so the smaller, current
    one paints on top; the outermost carries the name, so the trunk arrows
    attach outside the whole stack.
    """
    out, sizes = [], [r for _a, _b, r in runs]
    for i, r in enumerate(sorted(sizes, reverse=True)):
        nm = prefix if i == 0 else "%s%s" % (prefix, "bcd"[i - 1])
        out.append(box(nm, anchor, "(%g,0,0)" % (dx if i == 0 else 0.0),
                       r * HSCALE, 0.45, "\\CutColor",
                       opacity=[0.16, 0.34, 0.80][min(i, 2)]))
        if i > 0:
            anchor_i = "(%s-west)" % prefix
        out.append("\\node[font=\\tiny, text=bcred, anchor=west] at "
                   "([xshift=1.6mm]%s-north) {$%d$};\n" % (nm, r))
        if i == 0:
            anchor = "(%s-west)" % prefix
    # the growth itself, and when it happens
    out.append("\\draw[bcred, line width=0.6pt, "
               "{Straight Barb[length=1.2mm]}-{Straight Barb[length=1.2mm]}]\n"
               "  ([xshift=-3.4mm]%sc-north) -- ([xshift=-3.4mm]%s-north);\n"
               % (prefix, prefix))
    lbl = "\\,{\\to}\\,".join(str(r) for r in sizes)
    eps = ", ".join(str(a) for a, _b, _r in runs)
    out.append("\\node[font=\\scriptsize, text=bcred, align=center] at "
               "(%s-south) [below=1.5mm] {the layer grows: $r(e) = %s$\\\\"
               "{\\tiny at epochs %s; at $r=32$ the gate is the identity}};\n"
               % (prefix, lbl, eps))
    return "".join(out)


def gate(name, anchor, dx, label):
    return ("\\node[cutop] (%s) at ([shift={(%g,0,0)}] %s) {%s};\n"
            % (name, dx, anchor.strip("()"), label))


def op(name, anchor, dx, label):
    return ("\\node[opnode] (%s) at ([shift={(%g,0,0)}] %s) {%s};\n"
            % (name, dx, anchor.strip("()"), label))


def stage_widths(r):
    """Feature-map side at each stage, given the map fed to the first stage."""
    return [max(1, r // (2 ** i)) for i in range(N_STAGES)]


def main():
    assert SHOWN_R in set(v for sched in RESOLUTIONS.values() for v in sched), \
        "%d is not a resolution any schedule uses" % SHOWN_R
    runs = schedule_runs()
    inp = [red for red in REDUCTIONS if red.startswith("input")]
    stem = [red for red in REDUCTIONS if red.startswith("stem")]

    out = []
    for row, (kind, reds) in enumerate((("input", inp), ("stem", stem))):
        y = -ROW_DY * row
        p = "r%d" % row
        # the float image, always full size on the way in
        out.append(box(p + "img", "(%g,%g,0)" % (ROW_X, y), "(0,0,0)",
                       FULL * HSCALE, 0.9, "\\ImgColor"))
        prev = "(%simg-east)" % p

        if kind == "input":
            out.append(growing_gate(p + "cut", prev, GATE_PITCH, runs))
            prev = "(%scut-east)" % p
            out.append(box(p + "img2", prev, "(%g,0,0)" % GATE_PITCH,
                           SHOWN_R * HSCALE, 0.9, "\\ImgColor"))
            prev = "(%simg2-east)" % p
            out.append(op(p + "norm", prev, PITCH, "$\\mathcal{N}$"))
            prev = "(%snorm.east)" % p
            stem_side = SHOWN_R
        else:
            out.append(op(p + "norm", prev, PITCH, "$\\mathcal{N}$"))
            prev = "(%snorm.east)" % p
            stem_side = FULL

        out.append(box(p + "stem", prev, "(%g,0,0)" % PITCH,
                       stem_side * HSCALE, 0.75, "\\ConvColor"))
        prev = "(%sstem-east)" % p

        gap_next = PITCH
        if kind == "stem":
            out.append(growing_gate(p + "cut", prev, GATE_PITCH, runs))
            prev = "(%scut-east)" % p
            gap_next = GATE_PITCH

        for st, side in enumerate(stage_widths(SHOWN_R)):
            nm = "%sst%d" % (p, st + 1)
            out.append(box(nm, prev, "(%g,0,0)" % gap_next, side * HSCALE,
                           0.75 + 0.35 * st, "\\BlockColor"))
            gap_next = PITCH
            out.append("\\node[font=\\scriptsize, text=black!70, align=center] "
                       "at (%s-south) [below=2.5mm] "
                       "{stage %d\\\\$%d{\\times}%d$};\n"
                       % (nm, st + 1, side, side))
            prev = "(%s-east)" % nm

        out.append(box(p + "gap", prev, "(%g,0,0)" % PITCH, 2.4, 1.0,
                       "\\NormColor"))
        out.append("\\node[font=\\scriptsize, text=black!70] at "
                   "(%sgap-south) [below=2.5mm] {GAP, fc};\n" % p)

        # arrows along the row
        chain = [p + "img"]
        if kind == "input":
            chain += [p + "cut", p + "img2", p + "norm", p + "stem"]
        else:
            chain += [p + "norm", p + "stem", p + "cut"]
        chain += ["%sst%d" % (p, i + 1) for i in range(N_STAGES)] + [p + "gap"]
        out.append("\\draw[connection] (%g,%g,0) -- node{\\midarrow} "
                   "(%simg-west);\n" % (ROW_X - 1.4, y, p))
        for a, b in zip(chain, chain[1:]):
            aa = "%s.east" % a if a.endswith("norm") else "%s-east" % a
            bb = "%s.west" % b if b.endswith("norm") else "%s-west" % b
            out.append("\\draw[connection] (%s) -- node{\\midarrow} (%s);\n"
                       % (aa, bb))
        out.append("\\draw[connection] (%sgap-east) -- node{\\midarrow} "
                   "++(1.3,0,0);\n" % p)

        # row caption: the short names at the left, the sentence under the row
        # so neither overhangs the panel's left edge
        names = " / ".join("\\texttt{%s}" % r.replace("_", "\\_") for r in reds)
        note = ("shrinks the float image before $\\mathcal{N}$, so every site "
                "-- the stem included -- sees $%d{\\times}%d$"
                % (SHOWN_R, SHOWN_R)) if kind == "input" \
            else ("keeps the input at $%d$ and shrinks the stem output instead, "
                  "so the stem still sees $%d{\\times}%d$ and only the blocks "
                  "are reduced" % (FULL, FULL, FULL))
        out.append("\\node[anchor=east, align=right, font=\\small] at "
                   "(%g,%g,0) {%s};\n" % (ROW_X - 1.6, y, names))
        out.append("\\node[anchor=west, font=\\scriptsize, text=black!80] at "
                   "(%g,%g) {%s};\n" % (ROW_X - 3.4, y - FULL * HSCALE * 0.28 / 2 - 1.05,
                                           note))

    OUT.write_text(
        "%% generated by scripts/fig_resolution_pnn.py -- do not edit\n"
        "%% Box pics: PlotNeuralNet (MIT), vendored in ../gaussian_sites_pnn/layers/\n"
        + "".join(out))
    print("illustrated reduction r =", SHOWN_R)
    print("stage sides at that r  :", stage_widths(SHOWN_R))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
