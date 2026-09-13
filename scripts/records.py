"""Read experiment records from the working tree **or** from a git ref.

Several batches live on teammates' branches rather than in this tree
(Idriss's ablation on ``continuation-gaussian-tv_exploration_1``, Aubin's
sigma0 sweep on ``adaptative-schedule``).  Copying them here would duplicate
large result directories; reading them from the pinned commit keeps a single
copy and makes every location reproducible:

    Location(ref=None,       path="results/kaggle_outputs/unified-j0-.../x.json")
    Location(ref="<commit>", path="results/kaggle_outputs/abl-j0-.../x.json")

A location with ``ref=None`` is read from the checked-out tree.  A location with
a ref is read with ``git cat-file`` from that exact commit, so the index keeps
working after the branch moves.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Location:
    path: str
    ref: str | None = None

    def as_dict(self) -> dict:
        return {"ref": self.ref, "path": self.path}

    def __str__(self) -> str:
        return self.path if self.ref is None else "%s:%s" % (self.ref[:10], self.path)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True,
                          text=True, encoding="utf-8").stdout


def resolve(ref: str) -> str:
    """Full commit hash of ``ref``; raises if the ref is unknown locally."""
    return git("rev-parse", "--verify", ref + "^{commit}").strip()


def list_files(prefix: str, ref: str | None = None) -> list[str]:
    """Every file under ``prefix`` (repository-relative, forward slashes)."""
    if ref is None:
        base = ROOT / prefix
        if base.is_file():
            return [prefix]
        return sorted(str(p.relative_to(ROOT)).replace("\\", "/")
                      for p in base.rglob("*") if p.is_file())
    out = git("ls-tree", "-r", "--name-only", ref, "--", prefix)
    return sorted(line for line in out.splitlines() if line)


class Reader:
    """Batched reads; one ``git cat-file --batch`` process per ref."""

    def __init__(self):
        self._procs: dict[str, subprocess.Popen] = {}

    def _proc(self, ref: str) -> subprocess.Popen:
        p = self._procs.get(ref)
        if p is None:
            p = subprocess.Popen(["git", "cat-file", "--batch"], cwd=ROOT,
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            self._procs[ref] = p
        return p

    def read_bytes(self, loc: Location) -> bytes:
        if loc.ref is None:
            return (ROOT / loc.path).read_bytes()
        p = self._proc(loc.ref)
        p.stdin.write(("%s:%s\n" % (loc.ref, loc.path)).encode("utf-8"))
        p.stdin.flush()
        header = p.stdout.readline().decode("utf-8").split()
        if len(header) < 3 or header[1] == "missing":
            raise FileNotFoundError(str(loc))
        size = int(header[2])
        data = p.stdout.read(size)
        p.stdout.read(1)                     # trailing newline
        return data

    def exists(self, loc: Location) -> bool:
        if loc.ref is None:
            return (ROOT / loc.path).is_file()
        try:
            self.read_bytes(loc)
            return True
        except FileNotFoundError:
            return False

    def read_json(self, loc: Location):
        # NaN/Infinity appear in diverged runs; json accepts them by default
        return json.loads(self.read_bytes(loc).decode("utf-8"))

    def close(self):
        for p in self._procs.values():
            p.stdin.close()
            p.wait()
        self._procs.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def introducing_commit(path: str, ref: str = "HEAD") -> str | None:
    """The commit on ``ref`` that first added ``path`` (a file or directory)."""
    out = git("log", ref, "--diff-filter=A", "--format=%H", "--reverse", "--", path)
    lines = out.split()
    return lines[0] if lines else None
