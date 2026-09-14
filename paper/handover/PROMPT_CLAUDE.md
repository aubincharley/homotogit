Please integrate and continue the attached AISTATS-style LaTeX working paper in our research repository. This is a report-writing task. Do not launch training or change the selected methods to make the draft easier to explain.

The project is intentionally incomplete: the introduction, related work and methodology have been drafted; the main transfer table, theoretical result and loss visualizations are reserved for the team's ongoing work. Keep Max's direct English style, with “we”, clear motivation and restrained claims. Avoid rewriting this into a generic machine-learning sales pitch or expanding it into a long survey.

## 1. Integrate the paper without disrupting the research branches

Read repository instructions and inspect the current branches and local changes. Create a dedicated manuscript branch/worktree from the appropriate up-to-date shared base, preferably `continuation-core`; preserve everyone's uncommitted work and do not merge the whole exploratory codebase into the core just to access figures. The organized benchmark can be read from its own branch/worktree at recorded commits.

Put the attached project under the repository's appropriate paper/report directory, preserving its modular organization. Keep `main.tex`, the supplied official `aistats2026.sty` and `fancyhdr.sty`, the bibliography, and the distinction between main text and single-column appendices. The style files are unchanged from Max's supplied pack. Use `preprint` for now; this is not an actual AISTATS 2026 submission. Author names/order come from Max's earlier report and should not be guessed or reordered.

Read `README.md`, `handover/SOURCES_AND_PENDING.md`, and `figures/README.md` before editing. The snapshot was written from the conversation and reports, not a fresh checkout of our code. Your repository evidence takes precedence over a stale description, with any material correction documented.

## 2. Verify the scientific content against the code and records

Keep exactly these three methods and the plain control:

- `resolution_max_b1`: progressive adaptive max-pooling after zero-based residual block 1.
- `gaussian_postrelu`: Gaussian-only continuation at the ten selected post-ReLU sites.
- `resolution_max_b1_gaussian_conv`: the same resolution intervention plus Gaussian continuation at 19 convolution outputs.
- `plain`: no temporary intervention.

The combined method's placement differs from the Gaussian-only method. Do not silently turn this into a 2×2 factorial design. Preserve the historical sigma scaling by `r/32` at all 19 sites, including the five upstream sites still at 32×32. Confirm that “after block 1” corresponds to the pre-hook on `blocks[2]` and explain it once in human-readable terms.

Complete Appendix A from the frozen presets: exact resolution/blur schedules, kernel support and normalization, padding, bypass behavior, sigma units/scaling and transition indexing. Verify all reported numbers and training details against the audited index and raw records. The existing abstract's 81.51±0.22 versus 75.43±0.76, and the selected 80.37±0.61 and 79.91±0.27, refer to the unified three-seed comparison. Update all occurrences together if the audit changes them.

Respect the corrected asset provenance: no “set C”, no universal batch offset, no claim that unified alone is paired. Pair using actual hashes and run conditions. Keep numerical divergences visible, one-seed SD as unavailable, reruns distinct from duplicate files, current/target endpoints explicit, and training-probe CE distinct from full-training CE. Do not carry older resbench runtimes into the unified combinations. The TV-budget previews and the later quadratic reconstruction operators are separate experiments.

## 3. Add the existing exploratory figures and complete the appendix

Reuse/adapt the audited plotting pipeline and regenerate vector PDF figures from records where needed. Use the filenames and insertion points in `figures/README.md`; split them further if that improves readability. Put all broad exploratory grids, operator sweeps, placement comparisons and schedule comparisons in the appendices. Add the complete audited numerical table there, grouped by experiment/provenance rather than one universal ranking.

Use clear labels, a black plain baseline and a consistent palette for the three retained methods. Plot actual seed summaries and identify whether bands are sample SD or confidence intervals. For final comparisons of the selected procedures use the common full-resolution, filter-free path. Historical current-path learning curves are fine when clearly labelled; any bypass/target diagnostic belongs in a separate panel. No unexplained dashed curves, invented trajectories for missing checkpoints, interpolation presented as observed data, or tiny 111-row screenshot.

Every figure/table must retain provenance to run IDs, commits, seeds and generator commands, in a sidecar file or source comments. Replace the placeholder captions with captions that describe the actual included results and their limitations. Do not imply that three exploratory seeds establish a unique winning operator.

## 4. Leave room for the work being done by the team

- Aubin and Alexandre own transfer benchmarks: datasets, architectures and optimizers. Fill `tables/transfer.tex` only from completed, verified runs. Its current rows are placeholders, not a fixed commitment to unsupported combinations. Keep “not yet available” distinct from “failed” and “not applicable”. Do not reuse the exploratory test-selected results as independent confirmation. Record how each non-ResNet architecture maps the insertion sites and how scales/resolutions transfer.
- Idriss owns the theory. Integrate his supplied model and result when available; otherwise keep the short question and explicit pending note. Do not manufacture a theorem, proof or PAC-Bayes bound to fill the space. An illustrative regression theorem is not a theorem about our CNN. A smaller risk upper bound alone cannot establish a lower true risk.
- Max owns visualization. Keep the three questions: intervention at fixed weights, trajectory changes, and sensitivity of final weights. Insert available analyses only. Shared axes, explicit BatchNorm/data policy, explained PCA variance and projection residuals are necessary to interpret the figures. A projected checkpoint can lie away from its true position in parameter space; its displayed background loss is not automatically its real loss. Never equate a flat 2D slice with a generalization proof.

Keep the main paper compact. It should eventually contain the focused transfer table, the key theory statement and one or two useful geometric figures; the historical search belongs in the appendix. Review the abstract/introduction/discussion when new evidence is integrated so completed and planned contributions remain distinguishable.

## 5. Compile, inspect and deliver

Compile with pdfLaTeX/BibTeX, resolve references, inspect the PDF visually and fix table/figure overflow. Preserve the official layout, author–year citations, two-column main text and single-column appendices. Keep the unfilled checklist out of this working draft; do not mark requirements as satisfied without evidence.

Commit the manuscript work on its dedicated branch and push that branch normally to the existing remote. Do not force-push, publish a preprint or submit anything externally. Return the branch/commit, compiled PDF, an Overleaf-ready ZIP, a short summary of edits and a precise list of items still waiting on the team. Do not launch new experiments as part of this task.
