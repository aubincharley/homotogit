.PHONY: build run quick baseline push status pull clean runs plot test \
        homotopy-quick homotopy act-quick act push-act explore

KERNEL = idrisselkhamlichi/cifar10-resnet
# Override if the venv is not activated: make quick PYTHON=.venv/bin/python
PYTHON ?= python3
DIR ?= runs/latest
CONFIG ?= homotopy_linear
ACT ?= act_linear
# One seed by default: push-act is a screen, not the 3-seed measurement.
# A trailing comment on this line would become part of the value.
SEEDS ?= 0

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

push-act:       ## push one arm as its own kernel, e.g. make push-act ACT=act_jump
	@$(PYTHON) -c "$$STAGE_ARM" $(ACT) "$(SEEDS)"
	$(PYTHON) build.py
	@cp dist/main.py .push/$(ACT)/main.py
	@echo "NOTE: a push resets the accelerator. Set GPU T4 x2 in the UI, then Save & Run All."
	kaggle kernels push -p .push/$(ACT)
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
# One kernel per arm, so the six can run and be pulled independently instead of
# overwriting each other's output. The arm's config and its overrides are
# stamped into main.py, because a script kernel gets no argv to carry them --
# then main.py is put back, so the working tree is never left holding one arm.
define STAGE_ARM
import json, pathlib, shutil, sys
arm, seeds = sys.argv[1], sys.argv[2].strip()
main = pathlib.Path("main.py")
pathlib.Path("main.py.orig").write_text(main.read_text())
text = main.read_text()
text = text.replace('KAGGLE_CONFIG = "baseline"', f'KAGGLE_CONFIG = "{arm}"')
text = text.replace("KAGGLE_ARGV = []", f'KAGGLE_ARGV = ["--seeds", "{seeds}"]')
main.write_text(text)

stage = pathlib.Path(".push") / arm
stage.mkdir(parents=True, exist_ok=True)
meta = json.loads(pathlib.Path("kernel-metadata.json").read_text())
meta["id"] = f"idrisselkhamlichi/cifar10-{arm.replace('_', '-')}"
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
