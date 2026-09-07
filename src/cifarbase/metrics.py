"""Loss and accuracy over a whole split.

One function, because a baseline needs one number and its spread across seeds,
not a panel of diagnostics. layer-l2-homotopy is where sharpness, Hessian traces
and memorisation live; adding them here would make the thing you fork heavier
than the question you forked it to ask.
"""
import torch
import torch.nn.functional as F

from cifarbase.data import NUM_CLASSES


@torch.no_grad()
def evaluate(model, split, batch_size=1000, per_class=False):
    """Mean cross-entropy and accuracy over `split`, un-augmented.

    Leaves the model in the mode it arrived in. Forgetting that is how an
    evaluation ends up silently updating BatchNorm's running statistics on the
    test set for the rest of training.
    """
    was_training = model.training
    model.eval()

    total_loss = 0.0
    correct = 0
    seen = 0
    hits = torch.zeros(NUM_CLASSES, dtype=torch.int64)
    counts = torch.zeros(NUM_CLASSES, dtype=torch.int64)

    for x, y in split.chunks(batch_size):
        logits = model(x)
        total_loss += float(F.cross_entropy(logits, y, reduction="sum"))
        predicted = logits.argmax(1)
        correct += int((predicted == y).sum())
        seen += y.numel()
        if per_class:
            hits += torch.bincount(y[predicted == y].cpu(), minlength=NUM_CLASSES)
            counts += torch.bincount(y.cpu(), minlength=NUM_CLASSES)

    if was_training:
        model.train()

    out = {"loss": total_loss / seen, "acc": correct / seen}
    if per_class:
        # A class with no examples would divide by zero; clamp rather than drop
        # it, so the list always has ten entries in label order.
        accs = (hits.float() / counts.clamp(min=1).float()).tolist()
        out["per_class"] = accs
        out["worst_class"] = int(min(range(NUM_CLASSES), key=lambda i: accs[i]))
        out["worst_class_acc"] = min(accs)
    return out
