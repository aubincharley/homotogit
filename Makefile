.PHONY: build run quick baseline push status pull clean runs plot

KERNEL = aubincharley/cifar10-resnet
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
