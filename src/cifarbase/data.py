"""Get CIFAR-10 from wherever it happens to be, then keep it resident on the GPU.

Resolution order, first hit wins:
  1. $CIFAR_DATA                          -- explicit override
  2. any /kaggle/input/** mount holding cifar-10-batches-py
  3. ./data                               -- local cache
  4. download cifar-10-python.tar.gz      -- needs enable_internet
  5. torchvision.datasets.CIFAR10         -- last resort

Kaggle gives about 4 vCPU. A torchvision DataLoader doing PIL crops on that many
cores would starve a T4 long before the GPU became the limit, so there is no
DataLoader here at all: the whole set sits on the device as uint8, the train
split pre-padded to 40x40 once (240 MB), and the "loader" is an index shuffle.
Normalisation and augmentation happen per batch, on the GPU, in a handful of ops.

Augmentation is PER SAMPLE. One crop offset shared across a batch is a much
weaker augmentation -- the network still sees the whole batch in identical
registration -- and it is a silent bug, in that it costs a point or so of test
accuracy without ever raising an error.
"""
import glob
import os
import pickle
import tarfile
import urllib.request

import torch

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
NUM_CLASSES = 10
PAD = 4
# Fixed, and deliberately unrelated to any run seed: see _corrupt_labels.
_NOISE_SEED = 20090601

_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
_TRAIN_BATCHES = [f"data_batch_{i}" for i in range(1, 6)]
_TEST_BATCH = "test_batch"


def _cache_dir():
    return "/kaggle/working/cifar10" if os.path.isdir("/kaggle/working") else "data"


def _find_root(root):
    """The directory actually holding the pickled batches, or None."""
    hits = glob.glob(os.path.join(root, "**", "data_batch_1"), recursive=True)
    return os.path.dirname(sorted(hits)[0]) if hits else None


def _read_batch(path):
    with open(path, "rb") as fh:
        entry = pickle.load(fh, encoding="latin1")
    x = torch.frombuffer(bytearray(entry["data"]), dtype=torch.uint8)
    x = x.view(-1, 3, 32, 32).clone()          # clone: frombuffer aliases the bytes
    y = torch.tensor(entry["labels"], dtype=torch.int64)
    return x, y


def _read_dir(root):
    xs, ys = zip(*[_read_batch(os.path.join(root, b)) for b in _TRAIN_BATCHES])
    test_x, test_y = _read_batch(os.path.join(root, _TEST_BATCH))
    return torch.cat(xs), torch.cat(ys), test_x, test_y


def _complete(archive):
    """Is this a whole tarball? An interrupted download leaves a plausible-looking
    file that only fails much later, and a cached truncated archive would then
    poison every run in the session."""
    try:
        with tarfile.open(archive) as tar:
            return any(name.endswith("data_batch_1") for name in tar.getnames())
    except Exception:                                       # noqa: BLE001
        return False


def _download(dest):
    os.makedirs(dest, exist_ok=True)
    archive = os.path.join(dest, "cifar-10-python.tar.gz")
    if os.path.exists(archive) and not _complete(archive):
        print("  cached archive is truncated; re-downloading")
        os.remove(archive)
    if not os.path.exists(archive):
        print(f"  downloading {_URL} (~170 MB)")
        # Download to .part and rename only once it verifies, so an interrupted
        # run never leaves a broken archive behind under the real name.
        partial = archive + ".part"
        urllib.request.urlretrieve(_URL, partial)
        if not _complete(partial):
            os.remove(partial)
            raise SystemExit(
                "!! the CIFAR-10 download finished truncated. Re-run, or mount a "
                "dataset holding cifar-10-batches-py and point CIFAR_DATA at it.")
        os.replace(partial, archive)
    with tarfile.open(archive) as tar:
        try:
            tar.extractall(dest, filter="data")            # Python 3.12+
        except TypeError:
            tar.extractall(dest)
    return dest


def _load_raw():
    for root in (os.environ.get("CIFAR_DATA"), "/kaggle/input", "data", _cache_dir()):
        if not root or not os.path.isdir(root):
            continue
        found = _find_root(root)
        if found:
            print(f"CIFAR-10: pickled batches under {found}")
            return _read_dir(found)

    print("CIFAR-10: nothing mounted, falling back to download")
    print("  /kaggle/input holds:", glob.glob("/kaggle/input/*") or "(nothing)")
    try:
        found = _find_root(_download(_cache_dir()))
        if found:
            return _read_dir(found)
    except Exception as exc:                                # noqa: BLE001
        print(f"  download failed ({exc}); trying torchvision")

    from torchvision import datasets
    train = datasets.CIFAR10(_cache_dir(), train=True, download=True)
    test = datasets.CIFAR10(_cache_dir(), train=False, download=True)

    def to_chw(array):
        return torch.from_numpy(array).permute(0, 3, 1, 2).contiguous()

    return (to_chw(train.data), torch.tensor(train.targets),
            to_chw(test.data), torch.tensor(test.targets))


class Split:
    """One split, resident on the device as uint8, sliced by an index tensor.

    Train splits are stored pre-padded to 40x40 so the random crop is a gather
    rather than a pad-per-batch. Everything else is stored at 32x32 and read out
    with a centre crop, which for an unpadded split is the identity.
    """

    def __init__(self, x, y, device, padded=False):
        self.raw = (torch.nn.functional.pad(x, (PAD,) * 4) if padded
                    else x).to(device).contiguous()
        self.y = y.to(device)
        self.device = device
        self.pad = PAD if padded else 0
        mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
        self._mean, self._std = mean * 255.0, std * 255.0

    def __len__(self):
        return self.y.numel()

    def subset(self, index):
        """A new Split over `index`, sharing this one's padding state.

        Not `Split(self.raw[index], ...)`: __init__ pads, and a train split is
        already padded, so rebuilding one that way would pad it twice and hand
        the model 48x48 rows. The cross-fit teacher needs complementary halves
        of the train split, which is the only caller.
        """
        out = object.__new__(type(self))
        out.raw = self.raw[index].contiguous()
        out.y = self.y[index]
        out.device = self.device
        out.pad = self.pad
        out._mean, out._std = self._mean, self._std
        return out

    def _view(self, index, augment=False, generator=None):
        """uint8 rows -> normalised float32 [B,3,32,32], cropped and maybe flipped."""
        raw = self.raw[index]
        if self.pad:
            if augment:
                raw = _random_crop(raw, self.pad, generator)
            else:
                raw = raw[:, :, self.pad:self.pad + 32, self.pad:self.pad + 32]
        x = (raw.float() - self._mean) / self._std
        if augment:
            flip = torch.rand(x.shape[0], 1, 1, 1, device=x.device,
                              generator=generator) < 0.5
            x = torch.where(flip, x.flip(-1), x)
        return x

    def _epoch_order(self, pool, needed, generator):
        """`needed` indices drawn from `pool`, reshuffled on every pass over it.

        A pool shorter than an epoch is walked more than once rather than
        ending the epoch early -- see `batches` for why that matters.
        """
        if pool.numel() == 0:
            raise ValueError("empty pool: an epoch cannot be filled from it")
        passes, have = [], 0
        while have < needed:
            perm = torch.randperm(pool.numel(), device=self.device,
                                  generator=generator)
            passes.append(pool[perm])
            have += pool.numel()
        return torch.cat(passes)[:needed]

    def batches(self, batch_size, generator=None, drop_last=True, augment=False,
                pool=None):
        """One epoch's worth of batches, shuffled.

        `pool` restricts the epoch to a subset of the split WITHOUT shortening
        it: an epoch is always a full split's worth of examples, so a curriculum
        that trains on half the data still takes the same number of SGD steps.
        Anything else would be a silent bug -- `lr_at` derives the whole
        schedule from `steps_per_epoch * epochs`, so an epoch that ran short
        would leave the cosine somewhere in the middle at the end of training,
        and the curriculum would be charged for the lr schedule's mistake.
        """
        if pool is None:
            order = torch.randperm(len(self), device=self.device,
                                   generator=generator)
        else:
            order = self._epoch_order(pool, len(self), generator)
        stop = len(self) - (len(self) % batch_size if drop_last else 0)
        for start in range(0, stop, batch_size):
            index = order[start:start + batch_size]
            yield self._view(index, augment, generator), self.y[index]

    def chunks(self, batch_size):
        """Every example once, in order, un-augmented. This is what evaluate reads."""
        for start in range(0, len(self), batch_size):
            index = slice(start, start + batch_size)
            yield self._view(index), self.y[index]


def _random_crop(raw, pad, generator=None):
    """Crop 32x32 out of a padded row at a PER-SAMPLE offset, with one gather."""
    n = raw.shape[0]
    size = raw.shape[-1] - 2 * pad
    dy = torch.randint(0, 2 * pad + 1, (n,), device=raw.device, generator=generator)
    dx = torch.randint(0, 2 * pad + 1, (n,), device=raw.device, generator=generator)
    rows = dy[:, None] + torch.arange(size, device=raw.device)          # n,size
    cols = dx[:, None] + torch.arange(size, device=raw.device)          # n,size
    batch = torch.arange(n, device=raw.device)[:, None, None]
    return raw[batch, :, rows[:, :, None], cols[:, None, :]].permute(0, 3, 1, 2)


def _noise_draw(n_train, frac):
    """The examples `_corrupt_labels` mislabels, and the generator behind them.

    Split out so `corrupted_mask` can regenerate exactly the same draw later in
    the run without anything having to be remembered or threaded through. One
    definition of the draw, used twice.
    """
    k = int(round(frac * n_train)) if frac > 0 else 0
    if k == 0:
        return None
    generator = torch.Generator().manual_seed(_NOISE_SEED)
    return torch.randperm(n_train, generator=generator)[:k], generator


def corrupted_mask(n_train, frac):
    """True where the label is one of the permuted ones.

    Asked for by `scoring.curriculum_order`, which reports how much of the easy
    pool is corrupted -- the one number that says whether sorting by margin
    actually separated the clean examples from the rest.
    """
    mask = torch.zeros(n_train, dtype=torch.bool)
    draw = _noise_draw(n_train, frac)
    if draw is not None:
        mask[draw[0]] = True
    return mask


def _corrupt_labels(y, n_train, frac):
    """Permute `frac` of the FIRST n_train labels, deterministically.

    Seeded from a constant rather than from the run's seed, so every arm and
    every seed trains against exactly the same corruption. The experiment varies
    the order in which examples arrive, never which labels are wrong -- if the
    corruption moved between arms, the comparison would be measuring the draw.

    Only the train portion is touched: the validation slice keeps its true
    labels, so model selection and every reported number stay honest. It is the
    objective that is meant to be hard, not the measurement.

    Why this exists at all: with clean CIFAR-10 the easy subproblem and the full
    problem are both well posed, so a curriculum connects two nearly identical
    points and has nothing to bridge. Corrupting the labels makes the target
    genuinely hard -- it can only be fitted by memorising contradictions -- while
    the easy end stays well posed. That is the gap a curriculum is for.
    """
    draw = _noise_draw(n_train, frac)
    if draw is None:
        return y
    victims, generator = draw
    y = y.clone()
    # A uniform redraw would leave about a tenth of them accidentally correct.
    # Shifting by 1..9 guarantees every victim really is mislabelled, which is
    # what keeps "margin < 0" lined up with "corrupted".
    shift = torch.randint(1, NUM_CLASSES, (victims.numel(),), generator=generator)
    y[victims] = (y[victims] + shift) % NUM_CLASSES
    print(f"label noise: {victims.numel()}/{n_train} train labels permuted "
          f"({frac:.0%}); val and test untouched")
    return y


def load_cifar10(device, cfg):
    """The three splits, already on `device`. Called once per run, not per seed."""
    train_x, train_y, test_x, test_y = _load_raw()
    if cfg["train_subset"]:
        train_x = train_x[:cfg["train_subset"]]
        train_y = train_y[:cfg["train_subset"]]

    # The validation split is the LAST val_size examples, deterministically, so
    # every seed trains and validates on exactly the same data. Seeding the split
    # would confound "which seed" with "which val set".
    #
    # The baseline runs with val_size=0: train on all 50k, report the final epoch
    # against the 10k test set, which is the convention published ResNet numbers
    # use. Set it above 0 and train.py selects on validation accuracy instead.
    val_size = min(int(cfg["val_size"]), len(train_y) - 1)
    cut = len(train_y) - val_size if val_size > 0 else len(train_y)

    train_y = _corrupt_labels(train_y, cut, float(cfg.get("label_noise", 0.0)))

    train = Split(train_x[:cut], train_y[:cut], device, padded=True)
    val = Split(train_x[cut:], train_y[cut:], device) if val_size > 0 else None
    test = Split(test_x, test_y, device)

    mb = sum(s.raw.element_size() * s.raw.nelement()
             for s in (train, val, test) if s is not None) / 1024 ** 2
    print(f"CIFAR-10: {len(train)} train / {len(val) if val else 0} val / "
          f"{len(test)} test, {mb:.0f} MB resident (uint8)")
    return train, val, test
