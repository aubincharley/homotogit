.PHONY: build study smoke quick baseline push status pull clean runs plot

KERNEL = alexandrecorrard/cifar-10-resnet-baseline
# Override if the venv is not activated: make quick PYTHON=.venv/bin/python
PYTHON ?= python3
DIR ?= runs/latest

build:          ## bundle src/ + main.py into dist/main.py
	$(PYTHON) build.py

study:          ## the campaign: 4 regimes x 4 arms x 1 seed (needs a GPU)
	$(PYTHON) main.py --study pacing_vs_order

smoke:          ## the SAME grid on a CPU: 16 arms, 4k images, 2-3 epochs
	CIFAR_ALLOW_CPU=1 $(PYTHON) main.py --study smoke

quick:          ## CPU plumbing check of the single-arm path, no grid
	CIFAR_ALLOW_CPU=1 $(PYTHON) main.py --config quick --study ""

baseline:       ## one arm, one seed -- for timing a change
	$(PYTHON) main.py --config baseline --study "" --seeds 0

push: build     ## rebuild, then push the kernel to Kaggle
	@echo "NOTE: this STARTS the run. machine_shape NvidiaTeslaT4 is honoured by the API,"
	@echo "      so it lands on a T4 without touching the UI. Watch it with 'make status'."
	kaggle kernels push -p .

status:         ## last kernel run status
	kaggle kernels status $(KERNEL)

pull:           ## fetch the finished run log, results.json and history.* into out/
	mkdir -p out && kaggle kernels output $(KERNEL) -p out/
	@echo "Then: $(PYTHON) analyze.py out/runs/<stamp>_pacing_vs_order"

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

plot:           ## plot a study dir (DIR=runs/latest), or any folder of runs
	$(PYTHON) analyze.py $(DIR)
