.PHONY: build run quick baseline push status pull clean runs plot test \
        homotopy-quick homotopy act-quick act push-act grdh push-grdh explore

KERNEL = idrisselkhamlichi/cifar10-resnet
# Override if the venv is not activated: make quick PYTHON=.venv/bin/python
PYTHON ?= python3
DIR ?= runs/latest
CONFIG ?= homotopy_linear
ACT ?= act_linear
GRDH ?= baseline50 act_linear50 act_anchor50 act_anchor_steps50
# One seed by default: push-act is a screen, not the 3-seed measurement.
# A trailing comment on this line would become part of the value.
SEEDS ?= 0
# The kernel id is cifar10-<arm>-<TAG>, and TAG defaults to the seeds (s0,
# s012). Set it when two protocols share a config and a seed list.
TAG ?=
FORCE ?=

build:          ## bundle src/ + main.py into dist/main.py
	$(PYTHON) build.py

run:            ## the reference recipe, all seeds (slow without a GPU)
	$(PYTHON) main.py

quick:          ## CPU smoke test: resnet18 w=16, 4k images, 2 epochs, 1 seed
	CIFAR_ALLOW_CPU=1 $(PYTHON) main.py --config quick

baseline:       ## the reference recipe, one seed -- for timing a change
	$(PYTHON) main.py --seeds 0

test:           ## unit tests: the homotopy gate and the landscape maths
	$(PYTHON) -m pytest tests/ -q

homotopy-quick: ## CPU smoke test of the homotopy path: does s move, do ckpts land
	CIFAR_ALLOW_CPU=1 $(PYTHON) main.py --config homotopy_quick --no-wandb

homotopy:       ## one arm of the pilot, e.g. make homotopy CONFIG=homotopy_linear
	$(PYTHON) main.py --config $(CONFIG)

act-quick:      ## CPU smoke test of the activation homotopy: does alpha move
	CIFAR_ALLOW_CPU=1 $(PYTHON) main.py --config act_quick --no-wandb

act:            ## one arm of the activation pilot, e.g. make act ACT=act_jump
	$(PYTHON) main.py --config $(ACT)

grdh:           ## the four GRDH arms, locally and in order
	@for arm in $(GRDH); do $(PYTHON) main.py --config $$arm || exit 1; done

push-grdh:      ## push the four GRDH arms as four kernels
	@for arm in $(GRDH); do $(MAKE) push-act ACT=$$arm PYTHON=$(PYTHON) || exit 1; done

push-act:       ## push one protocol as its own kernel, e.g. make push-act ACT=act_jump SEEDS=0,1,2
	@$(PYTHON) -c "$$STAGE_ARM" $(ACT) "$(SEEDS)" "$(TAG)" "$(FORCE)"
	$(PYTHON) build.py
	@cp dist/main.py .push/$(ACT)-$${TAG:-s$$(echo $(SEEDS) | tr -d ,)}/main.py
	@echo "NOTE: a push resets the accelerator. Set GPU T4 x2 in the UI, then Save & Run All."
	kaggle kernels push -p .push/$(ACT)-$${TAG:-s$$(echo $(SEEDS) | tr -d ,)}
	@$(PYTHON) -c "$$RESTORE_MAIN"

explore:        ## loss-landscape figures for a run trained with ckpt_every > 0
	$(PYTHON) explore.py $(DIR)

push: build     ## rebuild, then push the kernel to Kaggle
	@echo "NOTE: a push resets the accelerator. Set GPU T4 x2 in the UI, then Save & Run All."
	kaggle kernels push -p .

status:         ## last kernel run status
	kaggle kernels status $(KERNEL)

pull:           ## fetch the finished run log, results.json and history.* into out/
	mkdir -p out && kaggle kernels output $(KERNEL) -p out/

clean:          ## drop generated artifacts -- deliberately NOT runs/
	rm -rf dist out
	find src -name __pycache__ -type d -exec rm -rf {} +

# Inline rather than a separate script: it is a listing, not a tool, and a
# shell for-loop with nested quoting around python -c was worse than this.
# One kernel per PROTOCOL, not per arm. A push overwrites the kernel it targets
# and Kaggle keeps no copy of the displaced run, so pushing the same config at a
# different seed count destroyed the earlier results -- they survived only
# because they had been pulled to out/ first. The kernel id therefore carries a
# tag derived from the seeds (s0, s012, ...), so a change of protocol lands in a
# new kernel and the old one keeps its output. TAG= overrides it for a protocol
# the seeds do not distinguish.
#
# It also refuses to push over a kernel that is currently running, because that
# restarts it from scratch -- which silently cost a full run once already.
# FORCE=1 is the deliberate override.
define STAGE_ARM
import json, pathlib, subprocess, sys
arm, seeds, tag, force = (sys.argv[1], sys.argv[2].strip(),
                          sys.argv[3].strip(), sys.argv[4].strip())
tag = tag or "s" + seeds.replace(",", "")
main = pathlib.Path("main.py")
pathlib.Path("main.py.orig").write_text(main.read_text())
text = main.read_text()
text = text.replace('KAGGLE_CONFIG = "baseline"', f'KAGGLE_CONFIG = "{arm}"')
text = text.replace("KAGGLE_ARGV = []", f'KAGGLE_ARGV = ["--seeds", "{seeds}"]')
main.write_text(text)

stage = pathlib.Path(".push") / f"{arm}-{tag}"
stage.mkdir(parents=True, exist_ok=True)
meta = json.loads(pathlib.Path("kernel-metadata.json").read_text())
meta["id"] = f"idrisselkhamlichi/cifar10-{arm.replace('_', '-')}-{tag}"

if not force:
    probe = subprocess.run(["kaggle", "kernels", "status", meta["id"]],
                           capture_output=True, text=True)
    if "RUNNING" in probe.stdout or "QUEUED" in probe.stdout:
        raise SystemExit(
            f"!! {meta['id']} is still running. Pushing would restart it from "
            f"scratch and lose the run in progress.\n"
            f"   Wait for it, or pass FORCE=1 to push anyway.")
meta["title"] = meta["id"].split("/")[-1]
meta["code_file"] = "main.py"
(stage / "kernel-metadata.json").write_text(json.dumps(meta, indent=4) + "\n")
print(f"staged {meta['id']}  config={arm}  seeds={seeds}")
endef
export STAGE_ARM

define RESTORE_MAIN
import pathlib
orig = pathlib.Path("main.py.orig")
pathlib.Path("main.py").write_text(orig.read_text())
orig.unlink()
endef
export RESTORE_MAIN

define LIST_RUNS
import glob, json, os
# islink skips runs/latest, which would otherwise list its target twice.
paths = [p for p in sorted(glob.glob("runs/*/results.json"))
         if not os.path.islink(os.path.dirname(p))]
if not paths:
    print("no runs yet")
for path in paths:
    d = json.load(open(path))
    cfg, stat = d["config"], d["stats"].get("test_acc_selected", {})
    acc = stat.get("mean", float("nan"))
    print(f"{os.path.dirname(path):<42} {cfg['arch']:<9} {cfg['epochs']:>4}ep  "
          f"n={stat.get('n', '?')}  acc={acc:.4f}")
endef
export LIST_RUNS

runs:           ## one line per run: which config, how many seeds, what accuracy
	@$(PYTHON) -c "$$LIST_RUNS"

plot:           ## plot the last run  (make plot DIR=runs to compare them all)
	$(PYTHON) analyze.py $(DIR)
