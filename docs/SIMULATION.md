# Physical history and undo

The CPU example works without GPU dependencies. Physical experiments require
a separate, compatible simulator installation and licensed assets.

## Recorded environment

These identify the development environment; they are not a universal lockfile.

| Component | Recorded version |
| --- | --- |
| Python | 3.11 |
| BEHAVIOR-1K / OmniGibson | `v3.9.3-post2`, commit `a8247a8cc1633fe1ca0cc66aa07243d46c64f155` |
| Isaac Sim | 5.1.0 |
| PyTorch | 2.7.0 with CUDA 12.8 |
| StanfordVL/cuRobo | commit `78612f45cef52c3fa0298de243a54cd7ca614414` |
| Warp | 1.12.0 |
| PyArrow | 21.0.0 |

Use the upstream [BEHAVIOR installation instructions](https://behavior.stanford.edu/getting_started/installation)
and the pinned [BEHAVIOR source](https://github.com/StanfordVL/BEHAVIOR-1K/tree/a8247a8cc1633fe1ca0cc66aa07243d46c64f155).
Obtain assets from the upstream provider under its terms; they are not bundled.
Each recipe names the required scene, object models and support models.

[`requirements/runtime-fixes.txt`](../requirements/runtime-fixes.txt) and
[`requirements/constraints-primitives.txt`](../requirements/constraints-primitives.txt)
record the local compatibility adjustments. They are not full environment
installers. The development installation had conflicting upstream dependency
declarations; physical runs worked in the recorded environment, but a clean
installation on another machine has not been certified.

The upstream mesh-transform compatibility patch is included:

```bash
git -C third_party/BEHAVIOR-1K apply --check \
  ../../patches/omnigibson-curobo-mesh-transform.patch
git -C third_party/BEHAVIOR-1K apply \
  ../../patches/omnigibson-curobo-mesh-transform.patch
```

Run this only against the matching checkout. Install the local package into
the simulator environment with `python -m pip install --no-deps -e .`.
Keep model-serving dependencies in a separate environment.

## Generate a history

```bash
OMNIGIBSON_HEADLESS=1 OMNIGIBSON_GPU_ID=0 \
task-undo episode --case cases/reading_001.json \
  --max-steps 30000 --timeout 3600 --output outputs/history
task-undo check-history outputs/history
```

The worker must complete every prescribed physical action and record a valid
history. A recipe or successful preview is not evidence of complete generation.
If this stage fails, keep the failed attempt and report it as a history-generation
failure; do not continue to undo.

## Bind an authored reference and execute

Before inspecting an undo result, bind the supplied reference to the actual
successful history:

```bash
task-undo-bind-labels --episode outputs/history \
  --reference cases/reading_001_undo.json --output outputs/private/labels.json

task-undo baseline --policy claim_aware --execution-mode guarded_goal_v1 \
  --restore-mode contact_refresh_v1 \
  --episode outputs/history --max-steps 12000 --timeout 2400 \
  --output outputs/undo

task-undo evaluate outputs/undo --labels outputs/private/labels.json
```

The binder validates the history and reference, then records hashes of the
actual state, checkpoint and public history. It does not infer the reference
from the rule prediction. The authored reference must match the chosen recipe;
it is not an independently annotated human judgment.

The executor receives no label path. Evaluation checks restoration, protected
poses/supports, recorded grasp states and empty hands. Its `joint_success`
describes the physical reference task; combine this with an independent goal
comparison through `evaluation/end_to_end.py` to obtain the paper's G/E/T/J.

All output directories and bound-label files must be new. Original checkpoints
may initialize undo, but subsequent task changes use normal actions, not
teleportation or rollback.

## Reproduction boundary

The repository includes current runtime code and selected recipes. It does not
include the paper's complete frozen runs, private labels or checkpoint archives.
Running these commands creates a new development attempt; it does not replay
the paper's reported trials. See [RESULTS.md](RESULTS.md).
