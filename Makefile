.PHONY: build run quick baseline selfcheck pilot pilot-check anchor resume \
        push status pull clean runs plot help

# The lambda sweep for stage 1. 0 is included deliberately: it is the free drift
# curve, and D_max is only interpretable as a fraction of it.
PILOT_LAMBDAS ?= 0 0.0001 0.001 0.01 0.1 1

# The slug Kaggle actually assigned. A kernel's slug comes from its TITLE on the
# first push, not from `id` in kernel-metadata.json -- `id` only routes pushes to
# a kernel that already exists. Renaming the title later does not move the slug.
KERNEL = aubincharley/cifar-10-anchored-l2-vs-baseline
# Override if the venv is not activated: make quick PYTHON=.venv/bin/python
PYTHON ?= python3
DIR ?= runs/latest

build:          ## bundle src/ + main.py into dist/main.py
	$(PYTHON) build.py

run:            ## the reference recipe, all seeds (slow without a GPU)
	$(PYTHON) main.py

quick:          ## CPU smoke test: resnet18 w=16, 4k images, 2 epochs, 1 seed
	CIFAR_ALLOW_CPU=1 $(PYTHON) main.py --config quick

baseline:       ## the reference recipe, one seed -- for timing a change
	$(PYTHON) main.py --seeds 0

# -- the anchored method: stage 0, stage 1, stage 3 ---------------------------
# Run them in this order. Each one is a gate on the next, and every failure they
# catch produces a run that looks entirely normal.

selfcheck:      ## stage 0: the anchor's correctness gates. Needs no dataset.
	CIFAR_ALLOW_CPU=1 $(PYTHON) main.py --selfcheck

pilot:          ## stage 1: fixed-lambda sweep. The go/no-go for the whole design.
	@for lam in $(PILOT_LAMBDAS); do \
		echo "=== pilot lambda=$$lam ==="; \
		$(PYTHON) main.py --config pilot --anchor-lambda $$lam || exit 1; \
	done
	@echo "now: make pilot-check"

pilot-check:    ## stage 2: monotone separation, then beta and D_max for anchor.yaml
	$(PYTHON) analyze.py runs --pilot

anchor:         ## stage 3: the method. Refuses until beta and D_max are set.
	$(PYTHON) main.py --config anchor

# Stage 3 is ~2.4 h per arm on a T4 and does not fit one Kaggle session. Split it
# with --stop-after-epoch, never by lowering --epochs: total_steps and therefore
# the whole cosine curve come from `epochs`, so a shorter declared run trains its
# first half on a different schedule. checkpoint.load refuses that resume.
resume:         ## continue a stopped run: make resume CKPT=runs/latest/ckpt_s0.pt
	@test -n "$(CKPT)" || { echo "usage: make resume CKPT=runs/<dir>/ckpt_s0.pt"; exit 1; }
	$(PYTHON) main.py --config anchor --resume $(CKPT)

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

help:           ## this list
	@grep -hE '^[a-z-]+:.*##' $(MAKEFILE_LIST) \
		| sed -E 's/:.*## /\t/' | expand -t18
