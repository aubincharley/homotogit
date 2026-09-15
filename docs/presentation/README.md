# Presentation deck

`index.html` is a standalone, self-contained page — nine plates, one figure per
claim, with the commentary and the limits next to each. Open it directly in a
browser; it needs no server and no build step.

    open docs/presentation/index.html

It is also published as an artifact:
<https://claude.ai/artifact/Grg5d6D2tgsjMk4oghzbYn>

[`SPEAKER_NOTES.md`](SPEAKER_NOTES.md) is the companion: what to say on each
plate, the question each one attracts, and the eight theoretical axes these
measurements call for — also published, at
<https://claude.ai/artifact/KHdmwx12uiCBqmtkF6czXC>

## Where its numbers come from

`data.json` is generated, never edited by hand:

    python tools/build_presentation_data.py

Per-method curves come from the 12 reference cells of the probe shards under
`results/kaggle_outputs/cc-probe-*`; the dial test and its scatter come from
`results/grid_report.json`. Four blocks are transcribed from the documents
rather than recomputed, and the script says which: the three-source accuracy
table and the BatchNorm 2x2 ([`CROSS_STUDY.md`](../CROSS_STUDY.md)), the
PAC-Bayes distances ([`RESULTS_GRID.md`](../RESULTS_GRID.md) section 5), and the
schedules ([`METHODS.md`](../METHODS.md)).

## The plates

| | claim |
|---|---|
| 01 | the operator reaches exact identity before the end; sigma is non-monotone for the combined method |
| 02 | the gain reproduces to 0.25 pp, and the curricula fit the training set *less* |
| 03 | the solutions are flatter; the BatchNorm gauge explains at most 2 %; 36/36 keep the sign across policies and splits |
| 04 | **the dissociation** — blur shifts the frequency radius, resolution reduction does not, and both gain ~17 % |
| 05 | no accessible linear regime, and the infinitesimal-versus-finite reversal in two spaces |
| 06 | no quantity acts as a dial; every strong pooled coefficient is a grouping artefact |
| 07 | every PAC-Bayes route tried is closed, including the displaced prior |
| 08 | what the two studies establish together, and what each corrects in the other |
| 09 | established / not established |

Every figure carries its own table under **Données**, so nothing on the page is
readable only by colour.
