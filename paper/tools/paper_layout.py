"""Publication layout for the audited records; called by build_assets.py.

Only presentation and aggregation of existing records are performed here.
No training, checkpoint evaluation, or interpolation of missing data is performed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from build_assets import M, LONG, COLOR, POLICY_SHORT, pm, tex, save, PAPER

# Every result row names the model, optimizer and epoch budget. Dataset sizes
# appear once per block; learning rates and schedule adaptations are in the text.
GROUPS = [
    (r"CIFAR-10: 50\,000 training / 10\,000 test images", [
        ("sgd", "ResNet-20 / BN / ReLU", "SGD", 30),
        ("adam", "ResNet-20 / BN / ReLU", "Adam", 30),
        ("adamw", "ResNet-20 / BN / ReLU", "AdamW", 30),
        ("radam", "ResNet-20 / BN / ReLU", "RAdam", 30),
        ("resnet20act_cifar10/gelu", "ResNet-20 / BN / GELU", "SGD", 30),
        ("resnet20act_cifar10/silu", "ResNet-20 / BN / SiLU", "SGD", 30),
        ("resnet20gn_cifar10", "ResNet-20 / GN / ReLU", "SGD", 30),
        ("vgg11_cifar10", "VGG-11 / BN / ReLU", "SGD", 30),
    ]),
    (r"SVHN: 50\,000 training / 26\,032 test images", [
        ("svhn", "ResNet-20 / BN / ReLU", "SGD", 30),
    ]),
    (r"STL-10: 5\,000 training / 8\,000 test images", [
        ("stl10", "ResNet-20 / BN / ReLU", "SGD", 60),
    ]),
    (r"CIFAR-10: 5\,000 training / 10\,000 test images", [
        ("cifar10_5k/stl_budget", "ResNet-20 / BN / ReLU", "SGD", 60),
        ("cifar10_5k/reference_updates", "ResNet-20 / BN / ReLU", "SGD", 293),
    ]),
]


def superior(a, b, plain):
    """Bold only the accuracy mean when it exceeds its own setting's baseline."""
    return (r"$\mathbf{%.2f}\pm%.2f$" if a > plain else r"$%.2f\pm%.2f$") % (a, b)


def panel(title, n):
    return r"\addlinespace[4pt]\multicolumn{%d}{@{}l@{}}{\textit{%s}}\\[2pt]" % (n, title)


def table(name, label, caption, cols, header, rows, star=False, foot=""):
    env = "table*" if star else "table"
    placement = "!htbp" if name in {"input_grid", "sensitivity_validation_paired"} else "!t"
    # Centre the headings over numerical columns while keeping values aligned.
    headings = header.split(" & ")
    assert len(headings) == len(cols)
    header = " & ".join(r"\multicolumn{1}{c}{%s}" % h if c == "r" else h
                        for h, c in zip(headings, cols))
    tabcolsep = "3.8pt" if name == "transfer_ce" else "4pt"
    body = (r"\begin{%s}[%s]" % (env, placement) + "\n" + r"\caption{%s}\label{%s}" % (caption, label)
            + "\n" + r"\centering\small\setlength{\tabcolsep}{%s}\renewcommand{\arraystretch}{1.08}" % tabcolsep
            + "\n" + r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}%s@{}}" % cols
            + "\n\\toprule\n" + header + "\\\\\n\\midrule\n" + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular*}\n")
    if foot:
        body += r"\par\smallskip\begin{minipage}{\linewidth}\footnotesize %s\end{minipage}" % foot + "\n"
    tex(name, body + r"\end{%s}" % env)


def benchmark_tables(S):
    rows, gains, costs, ce = [], [], [], []
    for title, settings in GROUPS:
        rows.append(panel(title, 7))
        gains.append(panel(title, 6))
        costs.append(panel(title, 8))
        ce_rows = []
        for st, model, opt, epochs in settings:
            g = S[S.setting == st].set_index("method")
            p = g.loc["plain"]
            prefix = f"{model} & {opt} & {epochs}"
            rows.append(prefix + " & " + " & ".join(
                superior(g.loc[m].test_acc_mean, g.loc[m].test_acc_sd, p.test_acc_mean)
                for m in M) + r"\\")
            gains.append(prefix + " & " + " & ".join(
                pm(g.loc[m].gain_mean, g.loc[m].gain_sd, sign=True) for m in M[1:]) + r"\\")
            costs.append(prefix + f" & {int(p.eval_every_epochs)} & " + " & ".join(
                pm(g.loc[m].wall_minutes_mean, g.loc[m].wall_minutes_sd) for m in M) + r"\\")
            if pd.notna(p.test_ce_mean):
                # All CE rows here use SGD, made explicit in the caption.
                ce_rows.append(f"{model} & Train & " + " & ".join(
                    ce_value(g.loc[m].train_probe_ce_mean, g.loc[m].train_probe_ce_sd) for m in M) + r"\\")
                ce_rows.append(r"\quad %d epochs & Test & " % epochs + " & ".join(
                    pm(g.loc[m].test_ce_mean, g.loc[m].test_ce_sd, 3) for m in M) + r"\\")
        if ce_rows:
            ce.append(panel(title, 6))
            ce.extend(ce_rows)

    table("transfer_main", "tab:transfer",
          r"Final test accuracy (\%), mean $\pm$ sample SD over three seeds. \textbf{Bold means exceed Plain}; this is a descriptive comparison, not a significance test. All runs use the final epoch, without augmentation. BN: BatchNorm; GN: GroupNorm (8 groups). R, G and RG are defined in Table~\ref{tab:methods}.",
          "llrrrrr", r"Model / normalization / activation & Optimizer & Epochs & Plain & R & G & RG", rows, star=True,
          foot=r"The first SGD row reuses the selection batch. Adaptive learning rates were selected using seed-0 test accuracy. Both 5k rows use the same training subset; 60 and 293 epochs correspond to 2\,400 and 11\,720 updates. Protocol details and paired gains are in Appendix~\ref{app:transfer}.")
    table("transfer_paired", "tab:transfer-paired",
          r"Paired accuracy gains over Plain (percentage points), mean $\pm$ sample SD over three seeds. Configurations and evaluation sets match Table~\ref{tab:transfer}; differences are formed within each setting and seed.",
          "llrrrr", r"Model / normalization / activation & Optimizer & Epochs & $\Delta$ R & $\Delta$ G & $\Delta$ RG", gains,
          foot=positive_pairs_note(S))
    table("costs", "tab:costs",
          r"Total wall time in minutes, mean $\pm$ sample SD over three seeds. Evaluation and checkpoint writes are included. The column $k$ means evaluation every $k$ epochs, plus the endpoint. Compare methods within a setting: GPU sessions and evaluation workloads differ across settings.",
          "llrrrrrr", r"Model / normalization / activation & Optimizer & Epochs & $k$ & Plain & R & G & RG", costs)
    table("transfer_ce", "tab:transfer-ce",
          r"Final cross-entropy (nats), mean $\pm$ sample SD over three seeds. All rows use SGD, learning rate $0.005$. Train uses a fixed 500-image probe; Test uses the complete test set. Adaptive-optimizer endpoint CEs were not recorded.",
          "llrrrr", r"Model / norm. / activation & Split & Plain & R & G & RG", ce)
    g = S[S.setting == "sgd"].set_index("method")
    rows = [LONG[m] + " & " + pm(g.loc[m].wall_minutes_mean, g.loc[m].wall_minutes_sd)
            + " & " + ("---" if m == "plain" else r"$%+.1f\%%$" % ((g.loc[m].wall_ratio_to_plain_mean - 1) * 100))
            + r"\\" for m in M]
    table("cost_main", "tab:cost-main",
          r"Training cost in the selection batch: CIFAR-10 (50k), ResNet-20-BN/ReLU, SGD, 30 epochs on a Tesla T4. Time includes evaluation each epoch and checkpoint writes; mean $\pm$ SD over three seeds. Extra time is the mean within-seed percentage change from Plain.",
          "lrr", r"Method & Time (min) & Extra time", rows)


def ce_value(a, b):
    if a < .01:
        d = max(0, 2 - int(np.floor(np.log10(a))))
        return pm(a, b, d)
    # Match the audited significant-digit convention without falsely printing
    # a small but nonzero dispersion as zero.
    d = max(0, 2 - int(np.floor(np.log10(a))))
    return pm(a, b, d)


def selection_table(settings, runs):
    g = settings[settings.setting == "sgd"].set_index("method")
    losses = runs[runs.setting == "sgd"].groupby("method").last_epoch_train_loss
    rows = []
    for m in M:
        r = g.loc[m]
        values = [superior(r.test_acc_mean, r.test_acc_sd, g.loc["plain"].test_acc_mean),
                  "---" if m == "plain" else pm(r.gain_mean, r.gain_sd, sign=True),
                  pm(r.test_ce_mean, r.test_ce_sd, 3),
                  pm(r.train_probe_ce_mean, r.train_probe_ce_sd, 3),
                  pm(losses.get_group(m).mean(), losses.get_group(m).std(), 3)]
        rows.append(LONG[m] + " & " + " & ".join(values) + r"\\")
    table("unified_reference", "tab:unified-reference",
          r"Endpoints of the final selection batch: CIFAR-10 / ResNet-20-BN / SGD, 30 epochs. Mean $\pm$ sample SD over three seeds; gains are paired to Plain. Test and 500-image training-probe CE use saved BatchNorm statistics. Last-epoch loss is the online mean while weights are updated. All losses are in nats; full-training-set CE was not evaluated in this batch. Bold accuracy means exceed Plain.",
          "lrrrrr", r"Method & Test acc. (\%) & Gain (pp) & Test CE & Probe CE & Last-epoch loss", rows)


def positive_pairs_note(S):
    labels = {"adam": "Adam", "adamw": "AdamW", "stl10": "STL-10"}
    names = dict(zip(M, ["Plain", "R", "G", "RG"]))
    exceptions = S[(S.method != "plain") & (S.n_seeds_gain_positive != 3)]
    assert set(exceptions.setting) <= set(labels), "Name any new exception explicitly."
    text = [r"%s with %s (%d/3)" % (names[r.method],labels[r.setting],r.n_seeds_gain_positive)
            for r in exceptions.itertuples()]
    return "Gains are positive in all three seed pairs except " + "; ".join(text) + "."


def landscape_tables(large, large_paired, hess, endpoints):
    rows, paired_rows = [], []
    for pol, name in [("saved", "Saved BatchNorm statistics"), ("recalibrated", "BatchNorm recalibrated at each point")]:
        rows.append(panel(name, 6))
        paired_rows.append(panel(name, 5))
        for st, pop in [("train_large", "Train subset (10k)"), ("test_full", "Full test set (10k)")]:
            for amp in [.1, .25]:
                g = large[(large.policy == pol) & (large.evaluation_set == st) & (large.amplitude == amp)].groupby("method").S.agg(["mean", "std"])
                rows.append(f"{pop} & {amp:g} & " + " & ".join(pm(g.loc[m,"mean"],g.loc[m,"std"],3) for m in M) + r"\\")
                g = large_paired[(large_paired.policy == pol) & (large_paired.evaluation_set == st) & (large_paired.amplitude == amp)].set_index("method")
                paired_rows.append(f"{pop} & {amp:g} & " + " & ".join(pm(g.loc[m].mean_delta,g.loc[m].sd_delta,3,True) + r"\,\textsuperscript{%d/5}" % int(g.loc[m].n_seeds_below_plain) for m in M[1:]) + r"\\")
    table("sensitivity_validation", "tab:sensitivity-validation",
          r"Finite sensitivity $S$ (nats) on larger evaluation sets. CIFAR-10, ResNet-20-BN/ReLU, SGD, final 30-epoch checkpoints of the five-seed landscape batch. Entries are means $\pm$ sample SD across seeds, each averaging 20 directions. $S$ is the centre-subtracted CE increase in Equation~\eqref{eq:sensitivity}, not the unperturbed loss.",
          "lrrrrr", r"Evaluation set & $\varepsilon$ & Plain & R & G & RG", rows)
    table("sensitivity_validation_paired", "tab:sensitivity-validation-paired",
          r"Paired differences $S_{\rm method}-S_{\rm plain}$ (nats), mean $\pm$ sample SD over five seeds. Superscripts count negative seed pairs. The populations and protocols are those of Table~\ref{tab:sensitivity-validation}.",
          "lrrrr", r"Evaluation set & $\varepsilon$ & $\Delta S$: R & $\Delta S$: G & $\Delta S$: RG", paired_rows)
    rows = []
    for m in M[1:]:
        values = []
        for probe in ["train_probe", "test_probe"]:
            w = hess[(hess.policy == "centre_frozen") & (hess.coords == "relative") & (hess.probe == probe)].pivot(index="seed",columns="method",values="top1")
            r = w[m] / w.plain
            values.append(pm(r.mean(),r.std()))
        rows.append(LONG[m] + " & " + " & ".join(values) + r"\\")
    table("hessian_main", "tab:hessian",
          r"Leading Hessian eigenvalue relative to Plain, mean $\pm$ SD of five within-seed ratios. CIFAR-10 / ResNet-20-BN / SGD, 30 epochs; 1k-image probes, centre-frozen BatchNorm and block-relative coordinates. Plain is 1; absolute values appear in Table~\ref{tab:hessian-all}.",
          "lrr", r"Method & Train probe & Test probe", rows)
    rows = []
    for coords, label in [("ordinary", "Ordinary parameter coordinates"), ("relative", "Block-relative parameter coordinates")]:
        rows.append(panel(label, 6))
        for pol in ["saved", "centre_frozen"]:
            for probe in ["train_probe", "test_probe"]:
                g = hess[(hess.policy == pol) & (hess.coords == coords) & (hess.probe == probe)]
                rows.append(POLICY_SHORT[pol] + " & " + ("Train" if probe == "train_probe" else "Test") + " & " + " & ".join(pm(g[g.method==m].top1.mean(),g[g.method==m].top1.std(),0) for m in M) + r"\\")
    table("hessian_all", "tab:hessian-all",
          r"Largest algebraic Hessian eigenvalue, mean $\pm$ sample SD over five seeds. CIFAR-10 / ResNet-20-BN / SGD, 30 epochs; CE on 1k-image probes and 268\,336 convolution/classifier weights. Coordinate groups define different matrices; compare methods within rows.",
          "llrrrr", r"BatchNorm policy & Probe & Plain & R & G & RG", rows)
    groups = endpoints.groupby("method")
    baseline = 100*groups.get_group("plain").test_full_acc.mean()
    rows=[]
    for m in M:
        g=groups.get_group(m)
        rows.append(LONG[m]+" & "+superior(100*g.test_full_acc.mean(),100*g.test_full_acc.std(),baseline)+" & "+pm(g.train_full_ce.mean(),g.train_full_ce.std(),3)+" & "+pm(g.test_full_ce.mean(),g.test_full_ce.std(),3)+r"\\")
    table("landscape_endpoints", "tab:landscape-endpoints",
          r"Endpoints of the separate five-seed landscape batch: CIFAR-10 / ResNet-20-BN / SGD, 30 epochs, saved BatchNorm. Accuracy uses all 10k test images; CE uses all 50k training or 10k test images. Mean $\pm$ sample SD. Bold accuracy means exceed Plain; this batch is not pooled with Table~\ref{tab:transfer}.",
          "lrrr", r"Method & Test accuracy (\%) & Train CE (nats) & Test CE (nats)", rows)


def input_tables(cells):
    # Keep the complete grid in Appendix E, with one
    # recipe heading per block rather than repeating it in every row.
    rows=[]
    variants=[("reference",r"Reference: $\eta=0.005$, $w=5\times10^{-4}$"),
              ("lr_low",r"Learning rate $\eta=0.0025$"),("lr_high",r"Learning rate $\eta=0.01$"),
              ("wd_low",r"Weight decay $w=10^{-4}$"),("wd_high",r"Weight decay $w=2\times10^{-3}$")]
    for variant,title in variants:
        rows.append(panel(title,7))
        subset=cells[cells.recipe_variant==variant]
        base=100*(1-subset[subset.method=="plain"].test_error).mean()
        for m in M:
            g=subset[subset.method==m];acc=100*(1-g.test_error).mean()
            a=(r"$\mathbf{%.2f}$" if acc>base else r"$%.2f$") % acc
            rows.append(LONG[m]+f" & {len(g)} & "+a+f" & ${g.train_ce.mean():.3f}$ & ${g.test_ce.mean():.3f}$ & ${g.jac_frobenius.mean():.1f}$ & "+r"$%s$" % f"{g.trace_deflated.mean():,.0f}".replace(",",r"\,")+r"\\")
    table("input_grid","tab:input-grid",
          r"All twenty input-diagnostic conditions: CIFAR-10 / ResNet-20-BN / SGD, 30 epochs. Each block changes one reference hyperparameter. Entries are means over $n$ seeds; changed recipes have one seed and no estimated SD. CE uses the full training/test sets; Jacobian norms use 1k training images. The all-parameter trace uses 5k images. Bold accuracy means exceed Plain within the same recipe.",
          "lrrrrrr",r"Method & $n$ & Test acc. (\%) & Train CE & Test CE & $\mathbb E\|J\|_F$ & $\widehat{\mathrm{tr}}(H)$",rows)
    ref=cells[cells.recipe_variant=="reference"]
    base=100*(1-ref[ref.method=="plain"].test_error).mean()
    rows=[]
    for m in M:
        g=ref[ref.method==m]
        rows.append(LONG[m]+" & "+superior(100*(1-g.test_error).mean(),100*g.test_error.std(),base)+" & "+pm(g.jac_frobenius.mean(),g.jac_frobenius.std(),1)+" & "+pm(g.jac_spectral.mean(),g.jac_spectral.std(),1)+" & "+pm(g.freq_mean_radius.mean(),g.freq_mean_radius.std())+r"\\")
    table("input_reference", "tab:input-reference",
          r"Input diagnostics in the separate reference grid: CIFAR-10 / ResNet-20-BN / SGD, 30 epochs, saved BatchNorm; mean $\pm$ sample SD over three seeds. Jacobian norms average 1k training images and use raw-intensity coordinates. Radius is in cycles per image. Bold accuracy means exceed Plain.",
          "lrrrr", r"Method & Test accuracy (\%) & $\mathbb E\|J\|_F$ & $\mathbb E\|J\|_2$ & Mean radius",rows)


def fig_sensitivity(paired):
    # Four small panels in one row keep the quantitative comparison in the body.
    for panels, name, size in [
        ([("saved","train_probe"),("saved","test_probe"),("recalibrated","train_probe"),("recalibrated","test_probe")],"sensitivity_summary",(7.0,2.25)),
        ([("centre_frozen","train_probe"),("centre_frozen","test_probe")],"sensitivity_centre_frozen",(6.6,2.35)),
    ]:
        fig,axes=plt.subplots(1,len(panels),figsize=size,layout="constrained")
        for ax,(pol,split) in zip(axes,panels):
            ax.axhline(0,color=".35",lw=.7)
            for m in M[1:]:
                g=paired[(paired.policy==pol)&(paired.split==split)&(paired.method==m)].sort_values("amplitude")
                x=g.amplitude.to_numpy();y=g.mean_delta.to_numpy();s=g.sd_delta.to_numpy()
                assert len(x)==10
                ax.plot(x,y,"o-",color=COLOR[m],ms=2.7,lw=1.2)
                ax.fill_between(x,y-s,y+s,color=COLOR[m],alpha=.14,lw=0)
            ax.set_xscale("log")
            ax.set_xticks([.005,.025,.1,.5],[".005",".025",".1",".5"])
            ax.set_xlabel(r"Amplitude $\varepsilon$",fontsize=8)
            ax.set_title(("Saved" if pol=="saved" else "Pointwise" if pol=="recalibrated" else "Centre-frozen")+" / "+("train" if split=="train_probe" else "test"),fontsize=9)
            ax.tick_params(labelsize=8)
            ax.grid(alpha=.18)
        axes[0].set_ylabel(r"$\Delta S$ (nats)",fontsize=9)
        h=[plt.Line2D([],[],color=COLOR[m],marker="o",ms=3,lw=1.2,label=LONG[m]) for m in M[1:]]
        fig.legend(handles=h,loc="outside lower center",ncol=3,frameon=False)
        save(fig,name)


def fig_surfaces(planes):
    from build_assets import fig_surfaces as full_surfaces
    full_surfaces(planes)  # A larger rendering remains among the source assets.
    g0=planes[(planes.seed==0)&(planes.grid==41)]
    matrices=[]
    for m in M:
        g=g0[g0.method==m]
        c=float(g[(g.a==0)&(g.b==0)].train_probe_ce.iloc[0])
        z=g.pivot(index="b",columns="a",values="train_probe_ce").sort_index().sort_index(axis=1)-c
        assert z.shape==(41,41) and not z.isna().any().any()
        matrices.append(z)
    lo=min(z.to_numpy().min() for z in matrices); hi=max(z.to_numpy().max() for z in matrices)
    norm=Normalize(lo,hi)
    fig=plt.figure(figsize=(7.0,2.6))
    names=["Plain", "Resolution (R)", "Gaussian (G)", "Combined (RG)"]
    for i,(m,z,name) in enumerate(zip(M,matrices,names)):
        ax=fig.add_axes([.012+.245*i,.15,.235,.77],projection="3d")
        xx,yy=np.meshgrid(z.columns,z.index)
        # Join all measured grid vertices; no fit, new evaluations, or smoothing.
        ax.plot_surface(xx,yy,z.to_numpy(),cmap="viridis",norm=norm,
                        rstride=1,cstride=1,linewidth=.16,
                        edgecolor=(0,0,0,.16),antialiased=True,shade=False)
        ax.set_xlim(-.5,.5);ax.set_ylim(-.5,.5);ax.set_zlim(lo,hi)
        ax.set_xticks([-.5,0,.5]);ax.set_yticks([0,.5]);ax.set_zticks([0,2,4])
        ax.set_xlabel("$a$",fontsize=8,labelpad=-7)
        ax.set_ylabel("$b$",fontsize=8,labelpad=-7)
        ax.tick_params(labelsize=6.5,pad=-3)
        ax.set_box_aspect((1,1,1))
        ax.view_init(elev=25,azim=-55)
        ax.set_title(name,fontsize=9,pad=4)
        for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
            axis.pane.fill=False
            axis._axinfo["grid"]["linewidth"]=.35
            axis._axinfo["grid"]["color"]=(.75,.75,.75,.5)
    cax=fig.add_axes([.30,.115,.40,.04])
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap="viridis"),cax=cax,
                    orientation="horizontal",ticks=[0,1,2,3,4])
    cb.set_label("Height and colour: CE above centre (nats)",fontsize=8,labelpad=1)
    cb.ax.tick_params(labelsize=7,pad=1,length=2)
    save(fig,"landscape_compact")
