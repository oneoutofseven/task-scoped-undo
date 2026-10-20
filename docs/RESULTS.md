# Working-paper results and release scope

The [six-page manuscript](../paper/manuscript.pdf) is a working draft, not an
accepted publication. This table is transcribed from its selected completed
English experiment versions. Each method has six conditions from three
physical ancestors.

| Method | Exact goal G | Predicted completion E | Physical task T | Joint G ∧ T |
| --- | ---: | ---: | ---: | ---: |
| Structured rule | 6/6 | 6/6 | 6/6 | 6/6 |
| Qwen direct | 1/6 | 4/6 | 1/6 | 0/6 |
| Qwen facts + rule | 6/6 | 6/6 | 6/6 | 6/6 |
| Phi-4 direct | 6/6 | 5/6 | 5/6 | 5/6 |
| Phi-4 facts + rule | 6/6 | 6/6 | 6/6 | 6/6 |

All six conditions of each selected method remain in the denominator, including
wrong goals and physical failures. Branches, translations and repeated runs
do not create independent physical ancestors. These small development results
do not establish a general model ranking.

A separate geometry-based history-generation study completed **10/16** histories
after **16/16** layout constructions and **16/16** initialization previews.
Those candidates had been exposed during development; this is not a fresh
held-out generalization test. It is a different cohort from the main table.

## What is included

- Layered runtime source, selected physical recipes and authored references.
- Synthetic English examples and CPU contract/regression tests.
- The paper PDF and an architecture illustration.
- Source hashes and release provenance in [`release.json`](../release.json).

## What is not included

- Complete raw trajectories, simulator checkpoints, model responses and
  per-run frozen source trees for the paper's historical experiments.
- Third-party simulator assets, simulator source and model weights.
- An independent human-labeling study or a completed held-out benchmark.

The public source is a curated development snapshot with small portability
wrappers. It must not be substituted for a historical experiment's frozen
source when claiming exact reproduction. The synthetic demo and mocked tests
are software checks, not additional model or physical successes.
