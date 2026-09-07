"""lambda_min(H) by Lanczos. The only cheap way to see a branch event coming.

An anchored homotopy traces a path of minimisers as lambda relaxes. That path can
bifurcate: a direction that was a strict minimum while the anchor was tight goes
indefinite as it loosens, and the run departs onto a different branch. From the
outside this is invisible -- the loss keeps falling, the accuracy keeps climbing,
and the only trace is that lambda_min(H) crossed zero.

Ten Lanczos iterations is not a spectrum. It is an extreme-eigenvalue probe, and
that is all that is wanted here: the SIGN of the smallest Ritz value and roughly
how far from zero it sits. Full reorthogonalisation, because at ten steps the
Krylov basis loses orthogonality fast and the cheap version silently returns
duplicated eigenvalues instead of the extremes.

The batch is the probe batch and the forward pass is the probe's, so lambda_min is
a deterministic function of w measured on exactly the same examples and the same
BatchNorm regime as d_t. Two diagnostics on two different functions of w would
not be comparable, and the whole point is to read them on one axis.
"""
import torch
import torch.nn.functional as F

from cifarbase.anchor import CONTAINER
from cifarbase.kernel import probe_context

# Fixed, and not derived from the run seed: every seed and every arm is measured
# from the same Krylov start, which is the same common-random-numbers argument
# that governs the drift tangents.
_START_SEED = 20260908


def _flat(tensors):
    return torch.cat([t.reshape(-1) for t in tensors])


def lanczos_extremes(model, probe_x, probe_y, iterations=10, deterministic=False):
    """(lambda_min, lambda_max) Ritz values of the loss Hessian at w.

    `iterations` is the HVP budget: one per iteration. Returns nan when the
    budget is under 2, which is what a disabled setting looks like rather than a
    crash halfway through a long run.
    """
    if iterations < 2:
        return {"hessian_lambda_min": float("nan"),
                "hessian_lambda_max": float("nan"),
                "hessian_iters": 0}

    params = [p for name, p in model.named_parameters()
              if p.requires_grad and not name.startswith(CONTAINER + ".")]

    with probe_context(model, deterministic):
        with torch.enable_grad():
            loss = F.cross_entropy(model(probe_x), probe_y)
            grads = torch.autograd.grad(loss, params, create_graph=True)

            def hvp(vector):
                split, offset = [], 0
                for param in params:
                    count = param.numel()
                    split.append(vector[offset:offset + count].view_as(param))
                    offset += count
                out = torch.autograd.grad(
                    sum((g * v).sum() for g, v in zip(grads, split, strict=True)),
                    params, retain_graph=True)
                return _flat([o.detach() for o in out])

            # A fixed SEED, not a fixed vector. Seeded means lambda_min stays a
            # deterministic function of w, so a change between two probes is the
            # network moving and not the Krylov space being redrawn. Random rather
            # than all-ones because a structured start vector can be very nearly
            # orthogonal to the extreme eigenvectors, and ten iterations is not
            # enough to recover from that -- the gate would report a Ritz value
            # from the middle of the spectrum as if it were the smallest.
            #
            # The basis is kept in the model's dtype: at ResNet-18 scale a float64
            # Lanczos vector is 90 MB and full reorthogonalisation needs all of
            # them resident, on a card already holding the dataset, w and w0. Ten
            # vectors is plenty of accuracy in float32; only the tridiagonal
            # algebra below needs float64.
            size = sum(p.numel() for p in params)
            generator = torch.Generator(device=probe_x.device).manual_seed(_START_SEED)
            q = torch.randn(size, generator=generator, device=probe_x.device,
                            dtype=probe_x.dtype)
            q /= q.norm()

            basis, alphas, betas = [q], [], []
            for step in range(iterations):
                w = hvp(basis[-1])
                alpha = float(w @ basis[-1])
                alphas.append(alpha)
                w = w - alpha * basis[-1]
                if step > 0:
                    w = w - betas[-1] * basis[-2]
                # Full reorthogonalisation. Ten vectors is 10 * 45 MB here, and
                # skipping it is how a 10-step Lanczos returns the same Ritz value
                # three times and no extreme at all.
                for vector in basis:
                    w = w - (w @ vector) * vector
                beta = float(w.norm())
                if beta < 1e-10 or step == iterations - 1:
                    break
                betas.append(beta)
                basis.append(w / beta)

    model.zero_grad(set_to_none=True)

    tri = torch.diag(torch.tensor(alphas, dtype=torch.float64))
    if betas:
        off = torch.tensor(betas[:len(alphas) - 1], dtype=torch.float64)
        tri += torch.diag(off, 1) + torch.diag(off, -1)
    ritz = torch.linalg.eigvalsh(tri)
    return {"hessian_lambda_min": float(ritz[0]),
            "hessian_lambda_max": float(ritz[-1]),
            "hessian_iters": len(alphas)}
