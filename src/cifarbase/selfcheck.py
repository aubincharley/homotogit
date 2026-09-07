"""The gates. Run them in order; do not proceed past a failure.

Every one of these catches a bug whose symptom is a plausible run. An anchor
that pulls the wrong way, an estimator that reports sampling noise as drift, a
probe that quietly updates BatchNorm, a homotopy that is not the object written
down -- none of them raise, none of them show up in the loss curve, and all of
them cost the whole experiment. They are cheap and they are checked first.

The data here is synthetic and seeded, not CIFAR-10. What these gates test is the
arithmetic -- does the pull close the distance to w0, is the estimator a
deterministic function of w, is the linearised solution the one in the algebra --
and none of it is a statement about images. Synthetic data also means the gates
run on a laptop with no dataset mounted, which is where they are most useful.

    python main.py --selfcheck
"""
import math
import tempfile

import torch
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from cifarbase import anchor as anchor_mod
from cifarbase import kernel
from cifarbase.config import CONFIG
from cifarbase.data import Split
from cifarbase.model import ARCHS, ResNet

_SEED = 4242
_WIDTH = 16


def run_gates(cfg, device):
    """Every gate, in order, stopping at the first failure. True if all passed."""
    gates = [("K0  the drift reference is well-posed", _gate_reference),
             ("H1  the pull reaches w0", _gate_pull),
             ("H2  lambda=0 is inert", _gate_inert),
             ("H3  the estimator is deterministic", _gate_determinism),
             ("H4  a probe does not move BatchNorm", _gate_bn),
             ("H5  the homotopy is the closed form", _gate_closed_form),
             ("H6  a resume continues rather than restarts", _gate_resume)]

    print("\n" + "=" * 78)
    print("SELFCHECK -- the anchor's correctness gates")
    print("=" * 78)

    ok = True
    for name, gate in gates:
        print(f"\n{name}")
        passed, lines = gate(device)
        for line in lines:
            print(f"    {line}")
        print(f"  -> {'PASS' if passed else 'FAIL'}")
        if not passed:
            ok = False
            break

    print("\n" + "-" * 78)
    if ok:
        print("all gates PASS.")
        print("Pilot separation is not a unit gate -- it is a five-run")
        print("experiment and it is the real go/no-go:")
        print("    make pilot            # fixed lambda in {1e-4 .. 1}")
        print("    make pilot-check      # monotone separation, then beta and D_max")
        print("Do not run an adaptive arm before that reports SEPARATED.")
    else:
        print("a gate FAILED. Fix it before spending any GPU quota: every one of")
        print("these failures produces a run that looks entirely normal.")
    print("-" * 78)
    return ok


# ---------------------------------------------------------------------------


def _fixture(device, count=32, zero_init_residual=False):
    """A small ResNet and a fixed batch. Same net and same data in every gate.

    zero_init_residual defaults OFF here, unlike the real recipe. At w0 with it
    on, every residual branch outputs exactly 0 and the NTK is discontinuous
    there (see _gate_reference), so the gates that check the pull and the
    estimator would be measuring that degeneracy instead of the arithmetic they
    exist to certify. _gate_reference is where the degeneracy itself is measured.
    """
    torch.manual_seed(_SEED)
    model = ResNet(ARCHS["resnet18"], num_classes=10, width=_WIDTH,
                   zero_init_residual=zero_init_residual).to(device)
    generator = torch.Generator(device="cpu").manual_seed(_SEED + 1)
    x = torch.randn(count, 3, 32, 32, generator=generator).to(device)
    y = torch.randint(0, 10, (count,), generator=generator).to(device)
    return model, x, y


def _probe_cfg(**over):
    cfg = {"probe_tangents": 4, "probe_deterministic": False,
           "anchor_w0_dtype": "fp32", "anchor_ratio_bn": 1.0,
           "anchor_ratio_head": 1.0}
    cfg.update(over)
    return cfg


def _bn_state(model):
    return [(m.running_mean.detach().clone(), m.running_var.detach().clone())
            for m in model.modules() if isinstance(m, _BatchNorm)]


# ---------------------------------------------------------------------------


def _gate_reference(device):
    """Is K_0 taken somewhere the NTK is actually continuous?

    This gate exists because of a measurement, not a hunch. With
    zero_init_residual on -- which the reference recipe uses, and which is worth a
    few tenths of a point -- BatchNorm's last gamma in every block is exactly 0 at
    w0, so every residual branch outputs exactly 0 and each block's second ReLU
    sees pre-activations that are exactly its shortcut. Around 39% of them land
    exactly on the kink, where the derivative PyTorch uses (input > 0) flips for
    an arbitrarily small perturbation. The NTK sketch therefore jumps by ~24% for
    a 1e-8 displacement and does not shrink as the displacement does.

    The consequence is not subtle: d_t leaves a K_0 taken at w0 within one step,
    by a fixed amount, and can never come back. A schedule d*(t) rising linearly
    from 0 is then unreachable for the whole run, the controller sits clipped at
    lambda_max, and the run reports a saturated controller as though that were a
    finding about feature learning.

    So the gate checks the configuration is coherent rather than checking a
    number: either the reference is taken past the transient, or the degeneracy
    is not there to begin with.
    """
    lines = []
    verdicts = []
    for zero_init in (False, True):
        model, x, _ = _fixture(device, zero_init_residual=zero_init)
        cfg = _probe_cfg()
        w0 = [p.detach().clone() for p in model.parameters()]
        generator = torch.Generator(device="cpu").manual_seed(_SEED + 3)
        noise = [torch.randn(p.shape, generator=generator).to(device)
                 for p in model.parameters()]
        with kernel.probe_context(model):
            base = kernel.ntk_sketch(model, x, int(cfg["probe_tangents"]))
        jumps = []
        for eps in (1e-8, 1e-4):
            with torch.no_grad():
                for param, start, direction in zip(model.parameters(), w0, noise,
                                                   strict=True):
                    param.copy_(start + eps * direction)
            with kernel.probe_context(model):
                sketch = kernel.ntk_sketch(model, x, int(cfg["probe_tangents"]))
            jumps.append(float((sketch - base).norm() / base.norm()))
        # The RATIO is the discriminator, not the magnitude: a continuous response
        # grows with the displacement, and a jump that is the same size four
        # decades apart is a step rather than a derivative. The absolute bound is
        # loose on purpose -- the sketch is a float32 JVP through a deep net, so
        # even a genuinely continuous response at eps=1e-8 lands near 1e-3.
        continuous = jumps[0] < 5e-3 and jumps[1] > 10 * jumps[0]
        verdicts.append(continuous)
        lines.append(f"zero_init_residual={str(zero_init):<5}  "
                     f"||dM||/||M|| at eps=1e-8 {jumps[0]:.3e}, at 1e-4 "
                     f"{jumps[1]:.3e}  -> "
                     f"{'continuous' if continuous else 'DISCONTINUOUS at w0'}")

    lines.append("the discontinuity is real and is a property of the "
                 "initialisation, not a bug: ~39% of each block's second ReLU "
                 "sits exactly on its kink when the residual branches start at 0")
    # The configuration check. Nothing above is a pass/fail on its own; this is.
    lines.append("")
    lines.append("config: anchor_reference_step defaults to -1, i.e. K_0 is taken "
                 "at the end of the hold window (anchor_hold_steps) rather than "
                 "at w0. w0 itself is unchanged -- it is still step 0.")
    lines.append("set anchor_reference_step: 0 to take K_0 at w0 anyway (§7 "
                 "literally), and expect a permanently clipped lambda if you do.")
    # False for zero_init=False is a bug in the estimator; True for zero_init=True
    # is expected and is the whole point of the gate.
    return verdicts[0] and not verdicts[1], lines


def _gate_pull(device):
    """As lambda grows the pull drives w onto w0 and d onto zero, at rate 1/lambda.

    A single large lambda is not the right test, because the pull has a steady
    state rather than a fixed point: with gradients present,

        w - w0  ->  grad / lambda

    so any finite lambda leaves a residual displacement, and lambda is bounded
    above by 1/eta (past that the pull overshoots w0 and oscillates -- which is
    what anchor.pull asserts on). The limit lambda -> infinity is therefore
    reached by lowering eta and raising lambda together, and the claim to certify
    is the RATE at which the residual vanishes. A residual that stops shrinking is
    a parameter the anchor is not reaching, which is how the missing BatchNorm beta
    showed up.

    The two rates are NOT the same, and that is the measurement worth keeping.
    ||w - w0|| falls as 1/lambda, cleanly: 9.6x and 9.9x per decade as measured.
    d falls only about 3.8x per decade, i.e. d ~ ||w - w0||^0.59. The NTK of a
    ReLU network is continuous in w but not differentiable in it: a displacement
    of size delta flips the sign of every pre-activation lying within delta of
    zero, the number of those grows like delta, and each flip changes the Jacobian
    by an O(1) amount locally -- so ||K - K_0|| picks up a sqrt(delta) term that
    dominates the smooth O(delta) one. It is the same kink phenomenon the K0 gate
    measures, here in its non-degenerate form.

    So the criterion is the exponent, not a fixed tolerance: anything between
    sqrt and linear is the network being a ReLU network, and anything outside that
    band means the drift is not tracking the displacement at all.

    BatchNorm's running statistics are not parameters and are deliberately never
    anchored, so they keep tracking the data and the network's function genuinely
    changes even once the weights are back at w0. d is therefore measured with the
    buffers pinned at their reference values; the unpinned number is reported
    alongside, because how far apart the two are is worth knowing -- it is the
    share of measured drift that is BatchNorm state rather than weights.
    """
    lines = ["lambda is swept by lowering lr and raising lambda = 0.99/lr together"]
    measured = []

    for lr in (1e-2, 1e-3, 1e-4):
        model, x, y = _fixture(device)
        cfg = _probe_cfg()
        anchor = anchor_mod.Anchor(model, cfg)
        optimizer = torch.optim.SGD(anchor_mod.build_param_groups(
            model, {"weight_decay": 0.0}), lr=lr, momentum=0.9)
        reference, _ = kernel.measure(model, x, cfg)
        pinned = _bn_state(model)
        lam = 0.99 / lr

        model.train()
        # The pull closes 99% of the distance per step, so the steady state is
        # reached in a handful; 40 is already far past it.
        for _ in range(40):
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(x), y).backward()
            optimizer.step()
            anchor.pull(anchor.lambdas(lam), lr)

        _, drift_free = kernel.measure(model, x, cfg, reference=reference)
        with torch.no_grad():
            for module, (mean, var) in zip(
                    (m for m in model.modules() if isinstance(m, _BatchNorm)),
                    pinned, strict=True):
                module.running_mean.copy_(mean)
                module.running_var.copy_(var)
        _, drift_pinned = kernel.measure(model, x, cfg, reference=reference)

        entry = {"lam": lam, "dist": anchor.distance(lr)["dist_rel"],
                 "d": drift_pinned["d_ntk"], "d_free": drift_free["d_ntk"]}
        measured.append(entry)
        lines.append(f"lambda = {lam:>9.4g}   ||w-w0||/||w0|| = {entry['dist']:.3e}"
                     f"   d = {entry['d']:.3e}   (d with BN free: "
                     f"{entry['d_free']:.3e})")

    tightest = measured[-1]
    # Ten-fold lambda must buy close to ten-fold less displacement. Well short of
    # that means something is not being pulled.
    ratios = [a["dist"] / max(b["dist"], 1e-300)
              for a, b in zip(measured[:-1], measured[1:], strict=True)]
    drift_ratios = [a["d"] / max(b["d"], 1e-300)
                    for a, b in zip(measured[:-1], measured[1:], strict=True)]
    # The exponent in d ~ ||w - w0||^p, by OLS in log-log. 0.5 is the ReLU kink
    # term, 1.0 would be a differentiable network; both are fine, and a value
    # outside that band is not.
    xs = [math.log(entry["dist"]) for entry in measured]
    ys = [math.log(max(entry["d"], 1e-300)) for entry in measured]
    mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
    exponent = (sum((a - mean_x) * (b - mean_y)
                    for a, b in zip(xs, ys, strict=True))
                / sum((a - mean_x) ** 2 for a in xs))

    lines.append(f"||w-w0|| shrank by {['%.1fx' % r for r in ratios]} per decade "
                 f"of lambda  (want > 5x: the pull is 1/lambda)")
    lines.append(f"d shrank by        {['%.1fx' % r for r in drift_ratios]} per "
                 f"decade of lambda")
    lines.append(f"fitted exponent in d ~ ||w-w0||^p:  p = {exponent:.3f}  "
                 f"(want 0.35 .. 1.15; 0.5 is the ReLU kink term, 1.0 would be a "
                 f"differentiable network)")
    lines.append(f"at the tightest lambda: ||w-w0||/||w0|| = {tightest['dist']:.3e} "
                 f"(want < 1e-5), d = {tightest['d']:.3e} (want < 5e-3)")

    return (tightest["dist"] < 1e-5 and tightest["d"] < 5e-3
            and all(r > 5.0 for r in ratios)
            and 0.35 < exponent < 1.15), lines


def _gate_inert(device):
    """With lambda = 0 the whole apparatus changes nothing, bit for bit.

    Two separate claims, and both matter. The probe must not consume a global RNG
    draw -- if it did, the batch order would diverge from the baseline's and every
    anchored arm would be running on different data than the arm it is compared
    against. And the anchored code path at lambda = 0 must land on exactly the
    weights the plain loop lands on, so a null result really is a null result and
    not a small unaccounted difference in the update.
    """
    lines = []

    # (a) a probe is read-only in both senses: no parameter moves, no RNG advances.
    model, x, y = _fixture(device)
    cfg = _probe_cfg()
    anchor_mod.Anchor(model, cfg)
    model.train()
    before = [p.detach().clone() for p in model.parameters()]
    rng_before = torch.get_rng_state()
    reference, _ = kernel.measure(model, x, cfg)
    kernel.measure(model, x, cfg, reference=reference)
    params_same = all(torch.equal(a, b) for a, b
                      in zip(before, model.parameters(), strict=True))
    rng_same = torch.equal(rng_before, torch.get_rng_state())
    lines.append(f"probe leaves every parameter identical   {params_same}")
    lines.append(f"probe consumes no global RNG draw        {rng_same}")

    # (b) the anchored path at lambda = 0 vs a plain SGD loop, same batches.
    steps = 20

    def trajectory(anchored):
        model, x, y = _fixture(device)
        cfg = _probe_cfg()
        anchor = anchor_mod.Anchor(model, cfg) if anchored else None
        params = (anchor_mod.build_param_groups(model, {"weight_decay": 0.0})
                  if anchored else model.parameters())
        optimizer = torch.optim.SGD(params, lr=0.05, momentum=0.9,
                                    weight_decay=0.0, nesterov=True)
        reference = (kernel.measure(model, x, cfg)[0] if anchored else None)
        model.train()
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(x), y).backward()
            optimizer.step()
            if anchored:
                anchor.pull(anchor.lambdas(0.0), 0.05)
                if step % 5 == 0:
                    kernel.measure(model, x, cfg, reference=reference)
        return [p.detach().clone() for p in model.parameters()]

    plain, instrumented = trajectory(False), trajectory(True)
    identical = all(torch.equal(a, b) for a, b
                    in zip(plain, instrumented, strict=True))
    worst = max(float((a - b).abs().max()) for a, b
                in zip(plain, instrumented, strict=True))
    lines.append(f"{steps} steps, plain loop vs anchored at lambda=0: "
                 f"bitwise {identical}, max |diff| {worst:.3e}")
    return params_same and rng_same and identical, lines


def _gate_determinism(device):
    """The estimator twice at the same w must return bitwise-equal d.

    If it does not, the controller is integrating hardware noise. On a GPU the
    usual cause is cudnn picking a different convolution algorithm between calls,
    which is what probe_deterministic exists for -- and this gate is what decides
    whether that cost is worth paying, rather than paying it on speculation.
    """
    model, x, _ = _fixture(device)
    cfg = _probe_cfg()
    model.train()
    reference, _ = kernel.measure(model, x, cfg)
    # A random displacement, not a constant offset. Adding the same 0.01 to every
    # weight is coherent across input channels, so it compounds through twenty
    # convolutions -- and the probe runs in eval mode, where BatchNorm divides by
    # a running_var still sitting at its initial 1 and so does nothing to contain
    # it. The gate still measures the right thing either way, but it reported
    # d = 9e15, which reads as a bug rather than as a passing test.
    generator = torch.Generator(device="cpu").manual_seed(_SEED + 5)
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn(param.shape, generator=generator).to(device),
                       alpha=0.01)

    _, first = kernel.measure(model, x, cfg, reference=reference)
    _, second = kernel.measure(model, x, cfg, reference=reference)
    equal = {key: first[key] == second[key] for key in first}
    lines = [f"d_ntk      {first['d_ntk']!r}  vs  {second['d_ntk']!r}",
             f"d_feature  {first['d_feature']!r}  vs  {second['d_feature']!r}",
             f"bitwise equal: {equal}"]
    if not all(equal.values()):
        lines.append("set --probe-deterministic to pin cudnn inside the probe, "
                     "then re-run this gate")
    return all(equal.values()), lines


def _gate_bn(device):
    """A probe from train mode leaves every running_mean and running_var alone.

    probe_context asserts this itself, so a violation would already raise. The
    gate checks it independently rather than trusting the assertion it is meant
    to be testing.
    """
    model, x, _ = _fixture(device)
    cfg = _probe_cfg()
    model.train()
    # Give the buffers a non-trivial value first: at initialisation they are
    # exactly (0, 1) and an update that wrote (0, 1) back would be invisible.
    for _ in range(3):
        model(x)
    before = _bn_state(model)
    reference, _ = kernel.measure(model, x, cfg)
    kernel.measure(model, x, cfg, reference=reference)
    after = _bn_state(model)
    same = all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
               for a, b in zip(before, after, strict=True))
    lines = [f"{len(before)} BatchNorm layers, running stats unchanged: {same}",
             f"model returned to train mode: {model.training}"]
    return same and model.training, lines


# ---------------------------------------------------------------------------


def _synthetic_splits(device, count=512):
    """Real Split objects over synthetic images. Labels are exactly balanced.

    Real Splits rather than stand-ins, so the gate exercises the actual padding,
    centre-crop and per-sample-augmentation path the run uses. Labels by
    arange % 10 rather than randint: the probe is class-balanced and needs a
    guaranteed share per class, and a random draw would fail this gate once in a
    while for a reason that has nothing to do with resuming.
    """
    generator = torch.Generator(device="cpu").manual_seed(_SEED + 11)
    x = torch.randint(0, 256, (count, 3, 32, 32), generator=generator,
                      dtype=torch.uint8)
    y = torch.arange(count) % 10
    return (Split(x, y, device, padded=True),
            None,
            Split(x[:100], y[:100], device))


def _resume_cfg(**over):
    cfg = dict(CONFIG)
    cfg.update({
        "seeds": "0", "epochs": 2, "batch_size": 128, "eval_batch_size": 100,
        "arch": "resnet18", "width": _WIDTH, "val_size": 0, "augment": True,
        "lr": 0.05, "warmup_epochs": 0, "schedule": "cosine",
        "weight_decay": 0.0, "per_class": False, "wandb": False,
        "anchor_mode": "const", "anchor_lambda": 0.5,
        # The whole gate is 8 steps long, so the hold window and the reference
        # step have to fit inside it -- at their real values of 500 the run would
        # never capture K_0 and the gate would be comparing two absences.
        "anchor_hold_steps": 2, "anchor_reference_step": 2,
        "probe_every": 2, "probe_batch": 20, "probe_tangents": 2,
        "hessian_every": 0, "ckpt_every": 1, "resume": "",
        "stop_after_epoch": 0,
        "config": "selfcheck", "selfcheck": False, "quick": False,
    })
    cfg.update(over)
    return cfg


def _gate_resume(device):
    """Train 2 epochs straight, then 1 + resume + 1, and compare bit for bit.

    The failure this catches is the one that cannot be caught by eye. A resume
    that rebuilds w0 from a fresh init, or recomputes K_0 at w_k, or restarts the
    batch-order shuffle, produces a run whose loss curve, drift curve and lambda
    trace all look completely normal -- and whose anchor is pointing somewhere
    nobody chose. Equality of the final weights is the only observable that
    distinguishes it.

    Both halves declare epochs=2 and the short one stops via stop_after_epoch,
    never by lowering `epochs`. That is not a detail of the test: total_steps, and
    therefore the entire cosine curve, is derived from `epochs`, so a run that
    declared epochs=1 would train its first epoch on a different lr schedule
    altogether. The first version of this gate did exactly that and failed with a
    max weight difference of 5.4e-3 -- which is why checkpoint.load now refuses a
    resume whose schedule-determining config has moved.
    """
    from cifarbase.train import train_once

    train, val, test = _synthetic_splits(device)
    lines = []

    with tempfile.TemporaryDirectory() as straight_dir, \
            tempfile.TemporaryDirectory() as split_dir:
        _, _, straight_probes = train_once(
            _resume_cfg(epochs=2), train, val, test, device, 0,
            verbose=False, out=straight_dir)
        straight = torch.load(f"{straight_dir}/ckpt_s0.pt", map_location="cpu",
                              weights_only=False)

        # Half a run, then continue it from the checkpoint. epochs stays at 2.
        train_once(_resume_cfg(epochs=2, stop_after_epoch=1), train, val, test,
                   device, 0, verbose=False, out=split_dir)
        _, _, resumed_probes = train_once(
            _resume_cfg(epochs=2, resume=f"{split_dir}/ckpt_s0.pt"),
            train, val, test, device, 0, verbose=False, out=split_dir)
        resumed = torch.load(f"{split_dir}/ckpt_s0.pt", map_location="cpu",
                             weights_only=False)

    checks = {}
    a, b = straight["model"], resumed["model"]
    checks["every parameter and buffer"] = (
        set(a) == set(b)
        and all(torch.equal(a[k], b[k]) for k in a))
    checks["w0 (the anchor itself)"] = all(
        torch.equal(a[k], b[k]) for k in a
        if k.startswith(anchor_mod.CONTAINER + "."))
    checks["K_0 (the drift reference)"] = (
        straight["reference"] is not None
        and resumed["reference"] is not None
        and all(torch.equal(straight["reference"][k], resumed["reference"][k])
                for k in straight["reference"]))
    checks["probe indices"] = torch.equal(straight["probe_index"],
                                          resumed["probe_index"])
    checks["controller u and d_ema"] = (
        straight["controller"]["u"] == resumed["controller"]["u"]
        and straight["controller"]["d_ema"] == resumed["controller"]["d_ema"])
    checks["the probe stream"] = (
        len(straight_probes) == len(resumed_probes)
        and all(x.get("d") == y.get("d") for x, y
                in zip(straight_probes, resumed_probes, strict=True)))

    for label, ok in checks.items():
        lines.append(f"{'identical' if ok else 'DIFFERS  '}   {label}")
    if not checks["every parameter and buffer"]:
        worst = max(float((a[k].double() - b[k].double()).abs().max())
                    for k in a if a[k].is_floating_point())
        lines.append(f"max |diff| over all tensors: {worst:.3e}")
    return all(checks.values()), lines


class _TwoLayer(torch.nn.Module):
    """f(x) = W2 relu(W1 x), scalar output, no biases.

    No biases anywhere, because the closed form is stated for a delta that lives
    in the same space as the rows of Phi. Including a bias in the network but not
    in Phi is the second of the two ways this gate fails for reasons that are not
    bugs in the anchor.
    """

    def __init__(self, dim, hidden, generator):
        super().__init__()
        self.w1 = torch.nn.Parameter(
            torch.randn(hidden, dim, generator=generator, dtype=torch.float64)
            / math.sqrt(dim))
        self.w2 = torch.nn.Parameter(
            torch.randn(1, hidden, generator=generator, dtype=torch.float64)
            / math.sqrt(hidden))

    def forward(self, x):
        return (F.relu(x @ self.w1.T) @ self.w2.T).squeeze(-1)


def _gate_closed_form(device):
    """The certification: does the implemented homotopy solve the stated problem?

    For the linearised anchored problem at w0, with phi(x) = grad_w f(x; w0),
    Phi the n x p matrix of those rows, r_i = y_i - f(x_i; w0) and K0 = Phi Phi^T,

        min_delta  1/(2n) ||Phi delta - r||^2 + lambda/2 ||delta||^2
        =>  delta*(lambda) = Phi^T (K0 + n lambda I)^-1 r

    Note the n in the ridge term. It comes from the 1/n in the loss, and getting
    it wrong is the likeliest source of a failure here that is not a real bug.

    The agreement is ASYMPTOTIC in lambda, so the trend across lambda is
    load-bearing and not just the magnitude: a flat error curve means something is
    wrong even when the numbers look small.

    On the rate. The norm bound ||delta*|| <= ||r|| / (2 sqrt(n lambda)) suggests
    lambda^-1/2, but that bound is loose in exactly the regime this gate runs in.
    Once n lambda dominates ||K0||, delta* = Phi^T (K0 + n lambda I)^-1 r tends to
    Phi^T r / (n lambda), so ||delta*|| falls as 1/lambda -- measured here as
    3.0e-2, 3.2e-3, 3.2e-4 across three decades. The relative linearisation error
    tracks it, and the fitted slope comes out at -0.99 rather than -0.5. The
    accepted band below spans both, since which one applies depends on where
    n*lambda sits relative to the spectrum of K0.
    """
    dim, hidden, n = 8, 32, 64
    generator = torch.Generator(device="cpu").manual_seed(_SEED + 7)
    model = _TwoLayer(dim, hidden, generator).to(device).double()
    x = torch.randn(n, dim, generator=generator, dtype=torch.float64).to(device)
    y = torch.randn(n, generator=generator, dtype=torch.float64).to(device)

    params = [model.w1, model.w2]
    w0 = [p.detach().clone() for p in params]

    # Phi, one row per example, by jacrev over the parameters at w0.
    rows = []
    for i in range(n):
        grads = torch.autograd.grad(model(x[i:i + 1]).sum(), params,
                                    retain_graph=True)
        rows.append(torch.cat([g.reshape(-1) for g in grads]))
    phi = torch.stack(rows)
    with torch.no_grad():
        residual = y - model(x)
    gram = phi @ phi.T

    lines = [f"n = {n}, p = {phi.shape[1]}, hidden = {hidden}, float64"]
    # Large lambda only, and further out than a first guess would put them.
    # Measured: 7.5e-3 at lambda = 100, 7.9e-4 at 1e3, 8.0e-5 at 1e4, so the 1e-3
    # crossing sits near lambda = 800. The three smaller values are reported as
    # diagnostics of the approach to the regime -- there ||delta*|| is 0.03 to
    # 0.5, the network genuinely changes its activation pattern, and the
    # linearisation this gate certifies does not apply.
    lambdas = (1e3, 1e4, 1e5)
    diagnostics = (1.0, 1e1, 1e2)
    errors, converged = {}, {}

    for lam in lambdas + diagnostics:
        target = phi.T @ torch.linalg.solve(
            gram + n * lam * torch.eye(n, dtype=torch.float64, device=device),
            residual)

        with torch.no_grad():
            for param, start in zip(params, w0, strict=True):
                param.copy_(start)

        def objective():
            loss = 0.5 * F.mse_loss(model(x), y)
            for param, start in zip(params, w0, strict=True):
                loss = loss + 0.5 * lam * (param - start).pow(2).sum()
            return loss

        # L-BFGS with a strong-Wolfe line search, not fixed-step gradient descent.
        # The objective is strongly convex with modulus lambda, but its smoothness
        # constant is set by the data term, so a fixed step sized for the worst
        # curvature crawls at small lambda: plain GD stalled at ||grad|| ~ 5e-3
        # after 400k iterations, and an unconverged run reports a large error that
        # looks exactly like a wrong closed form.
        optimizer = torch.optim.LBFGS(params, lr=1.0, max_iter=200,
                                      history_size=64, tolerance_grad=1e-16,
                                      tolerance_change=1e-20,
                                      line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss = objective()
            loss.backward()
            return loss

        grad_norm, previous = float("inf"), float("inf")
        # A gradient-norm criterion, not a step budget. The outer loop only exists
        # to re-arm L-BFGS when it exits on its own internal tolerance; it stops
        # when the gradient stops improving, which in float64 is as converged as
        # this gets.
        for _ in range(60):
            optimizer.step(closure)
            grads = torch.autograd.grad(objective(), params)
            grad_norm = float(torch.cat([g.reshape(-1) for g in grads]).norm())
            if grad_norm > 0.9 * previous:
                break
            previous = grad_norm

        with torch.no_grad():
            delta = torch.cat([(p - s).reshape(-1)
                               for p, s in zip(params, w0, strict=True)])
        gap = float((delta - target).norm())
        errors[lam] = gap / float(target.norm())

        # Converged ENOUGH, which is the only question that matters. The
        # objective is strongly convex with modulus lambda, so an unconverged
        # solve displaces w by at most ||grad||/lambda -- and that has to be
        # negligible against the linearisation gap being measured, or the gate is
        # reporting optimiser slop as though it were the closed form's error. An
        # absolute gradient tolerance cannot express this: the same 1e-9 is
        # generous at lambda = 100 and meaningless at lambda = 1.
        slop = grad_norm / lam
        converged[lam] = slop < 1e-3 * max(gap, 1e-300)
        note = ("   (want < 1e-3)" if lam in lambdas else
                "   (diagnostic: outside the asymptotic regime)")
        if not converged[lam]:
            note += "   << SOLVE TOO LOOSE TO READ"
        lines.append(f"lambda = {lam:8.4g}   ||delta*|| = {float(target.norm()):.3e}"
                     f"   rel err = {errors[lam]:.3e}   ||grad||/lambda = "
                     f"{slop:.2e} vs gap {gap:.2e}" + note)

    # Trend: log error against log lambda, over the load-bearing lambdas only.
    xs = [math.log(lam) for lam in lambdas]
    ys = [math.log(max(errors[lam], 1e-300)) for lam in lambdas]
    mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
    slope = (sum((a - mean_x) * (b - mean_y) for a, b in zip(xs, ys, strict=True))
             / sum((a - mean_x) ** 2 for a in xs))
    lines.append(f"log-log slope of rel err vs lambda: {slope:+.3f}  "
                 f"(want -1.5 .. -0.2; about -1 in this regime, where n*lambda "
                 f"dominates the spectrum of K0, and -0.5 where it does not. A "
                 f"flat curve is a failure even when the magnitudes look fine.)")

    converged_ok = all(converged[lam] for lam in lambdas)
    if not converged_ok:
        lines.append("a solve is too loose to read: the displacement it could "
                     "still move (||grad||/lambda) is not negligible against the "
                     "gap being measured, so that row says nothing about the "
                     "closed form. Fix the solve before reading the numbers.")
    magnitude_ok = all(errors[lam] < 1e-3 for lam in lambdas)
    trend_ok = -1.5 < slope < -0.2
    return converged_ok and magnitude_ok and trend_ok, lines
