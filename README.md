# Task-Scoped Physical Undo

**Undo one past task while preserving valid changes made afterwards.**

[GitHub repository](https://github.com/oneoutofseven/task-scoped-undo)

A robot sets out a cup and a book. Someone later reserves the book for
personal use. Cancelling the setup should return the cup and leave the book
alone. If the later claim instead concerns the cup, the correct undo changes,
even though the current scene and cancellation request stay the same.

[Task specification](docs/TASK_SPEC.md) · [Architecture](docs/ARCHITECTURE.md) ·
[Simulation guide](docs/SIMULATION.md) · [Working paper](paper/manuscript.pdf) ·
[中文说明](README.zh-CN.md)

![History evidence, goal inference, physical execution and independent evaluation](docs/assets/architecture.png)

## Quick start: no GPU or model required

Use Python 3.11 or newer. From this checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python examples/history_contrast.py
```

The example runs the actual rule implementation on **authored, synthetic
facts**, with the same current supports and request in both branches:

| Later activity | Restore | Protect |
| --- | --- | --- |
| Book adopted | Cup → storage | Book, plate |
| Cup adopted | Book → storage | Cup, plate |

This demonstrates the task boundary. It does not call a language model, execute
physics, or count as an experimental result.

```bash
python -m unittest discover -s tests -v
task-undo --help
```

The included tests use synthetic fixtures and mock model responses. They run
without a simulator or model service.

## Language-model goal inference

Configure an existing Chat Completions endpoint in
[`configs/qwen3vl.json`](configs/qwen3vl.json). Credentials, if needed, are read
from the environment variable named by `api_key_env`.

```bash
task-undo-predict --method direct \
  --input examples/book_adopted.json \
  --config configs/qwen3vl.json --output outputs/direct

task-undo-predict --method facts-rule \
  --input examples/book_adopted.json \
  --config configs/qwen3vl.json --output outputs/facts_rule
```

Both methods use the existing two-call table pipeline. Direct inference writes
`goal.json`. Facts + rule writes `rule_goal.json`, using model-extracted events
and the model-selected task; it also saves the shared model readout separately.
Requests, responses, failures and a source snapshot are retained. Output
directories must be new. See [METHODS.md](docs/METHODS.md).

## Physical execution

The simulator backend uses **OmniGibson / BEHAVIOR-1K, Tiago and cuRobo**.
Install those dependencies and assets separately, following the
[simulation guide](docs/SIMULATION.md). The core package does not install
the GPU stack.

```bash
task-undo episode --case cases/reading_001.json --output outputs/history
task-undo check-history outputs/history
task-undo-bind-labels --episode outputs/history \
  --reference cases/reading_001_undo.json --output outputs/private/labels.json
task-undo baseline --policy claim_aware --execution-mode guarded_goal_v1 \
  --restore-mode contact_refresh_v1 \
  --episode outputs/history --output outputs/undo
task-undo evaluate outputs/undo --labels outputs/private/labels.json
```

Continue only after the preceding command succeeds. The reference is a supplied
author judgment, bound to that exact generated history; it is used only for
offline scoring. A controller's completion flag alone does not establish
correct physical undo.

## Repository layout

```text
src/task_undo/
  domain/        Task semantics, restore/protect contracts and constraints
  agents/        Model clients, fact extraction and goal readouts
  application/   CLI, orchestration and shared execution
  simulation/    Navigation, grasping, placement and observation
  evaluation/    Independent trace scoring and provenance checks
  common/        Storage, paths, language checks and source snapshots
examples/        Small English inputs and the CPU contrast demo
cases/           Selected physical recipes and authored reference goals
tests/           CPU regression tests
docs/            Task, methods, architecture, simulation and results
paper/           Six-page working manuscript
```

## Scope and evidence

This is a compact research-code release. The task currently covers reversible
placements with sequential tasks, explicit persistent use claims and no outside
activity during undo. The rule is a specification baseline.

The working-paper results are summarized in [RESULTS.md](docs/RESULTS.md).
They come from separately frozen development experiments. This checkout is
not a complete raw-data release or a reproduction of every historical run;
private run archives, checkpoint files, model weights and third-party assets
are not bundled. Independent held-out generalization remains unverified.

[`release.json`](release.json) records the source and paper provenance.
