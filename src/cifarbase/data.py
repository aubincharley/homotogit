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

    def batches(self, batch_size, generator=None, drop_last=True, augment=False):
        order = torch.randperm(len(self), device=self.device, generator=generator)
        stop = len(self) - (len(self) % batch_size if drop_last else 0)
        for start in range(0, stop, batch_size):
            index = order[start:start + batch_size]
            yield self._view(index, augment, generator), self.y[index]

    def chunks(self, batch_size):
        """Every example once, in order, un-augmented. This is what evaluate reads."""
        for start in range(0, len(self), batch_size):
            index = slice(start, start + batch_size)
            yield self._view(index), self.y[index]

    def take(self, index):
        """The rows at `index`, un-augmented. The probe batch is read through here.

        Un-augmented, always: a probe under random crops would measure the crop
        rather than the network, and the drift curve would be pure noise. For a
        padded split "un-augmented" is the deterministic centre crop, which is the
        same view evaluate() sees.
        """
        return self._view(index), self.y[index]


def class_balanced_indices(split, count, seed):
    """`count` indices into `split`, count/NUM_CLASSES per class, drawn once.

    Seeded from a CONSTANT, never from the run seed, so every seed and every arm
    measures drift on the same images. A probe set that moved with the seed would
    put the batch's own difficulty into the across-seed spread of d_t, which is
    the one number the controller is steering by.

    Balanced because an unbalanced probe makes ||K_t - K_0|| partly a statement
    about which classes happen to be over-represented in it.
    """
    if count % NUM_CLASSES:
        raise ValueError(f"probe_batch {count} is not a multiple of "
                         f"{NUM_CLASSES}: a balanced probe needs an equal share "
                         f"per class")
    per_class = count // NUM_CLASSES
    generator = torch.Generator().manual_seed(int(seed))
    labels = split.y.to("cpu")
    picked = []
    for label in range(NUM_CLASSES):
        members = (labels == label).nonzero(as_tuple=True)[0]
        if members.numel() < per_class:
            raise ValueError(f"class {label} has {members.numel()} examples in "
                             f"this split but the probe wants {per_class}: lower "
                             f"probe_batch or raise train_subset")
        order = torch.randperm(members.numel(), generator=generator)
        picked.append(members[order[:per_class]])
    # Sorted so the probe batch's row order is a property of the split, not of the
    # order the classes were walked in -- K_t is only comparable to K_0 row by row.
    return torch.cat(picked).sort().values.to(split.device)


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

    train = Split(train_x[:cut], train_y[:cut], device, padded=True)
    val = Split(train_x[cut:], train_y[cut:], device) if val_size > 0 else None
    test = Split(test_x, test_y, device)

    mb = sum(s.raw.element_size() * s.raw.nelement()
             for s in (train, val, test) if s is not None) / 1024 ** 2
    print(f"CIFAR-10: {len(train)} train / {len(val) if val else 0} val / "
          f"{len(test)} test, {mb:.0f} MB resident (uint8)")
    return train, val, test
