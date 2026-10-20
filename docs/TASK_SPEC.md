# Task-Scoped Physical Undo: operational specification

This specification defines the placement-only task implemented in this repository. The released examples are synthetic demonstrations; experiment scope and the working-paper results are documented separately in [RESULTS.md](RESULTS.md).

## Inputs and scope

An episode provides a complete history of sequential, completed tasks, the current observation, and a request identifying one completed task to cancel. Object identities and support identities are fixed. Successful placement events and explicit, persistent use declarations constitute the evidence. All scored objects are registered in advance. Other room objects are outside the protection metric.

Tasks operate on reversible object placements. External placements and use declarations remain valid until undo finishes. There is no concurrent external activity during undo. Nested cancellation, ambiguous temporary moves, expired declarations, occupied restoration destinations, cabinet states, and consumption are excluded. Human activities and declarations are scripted; no human intent recognition or independent human annotation is claimed.

Let `b(t, o)` be the support of object `o` immediately before the cancelled task `t` begins. For each registered object:

1. If `t` never successfully placed it, protect it.
2. If another task placed it or explicitly claimed its use after the last placement by `t`, protect it.
3. Otherwise restore it to `b(t, o)`.

The cancelled task's own declarations do not survive cancellation. Repeated placements inside a task are processed in order, but restoration refers to the boundary before the entire task. Intermediate revisits do not introduce a new boundary. The present empirical design excludes a target task whose final object support equals its own starting support. Earlier independent tasks may make `b(t, o)` different from the initial episode support.

Every object belongs to exactly one of the restore or protect sets. This is a normative specification for the benchmark, rather than a claim that all human activities follow these rules. The rule implementation is a specification baseline. Independent forward-ledger code and authored labels check implementation consistency; neither replaces independent human assessment.

## Execution and scoring

A saved simulator state may initialize undo. Once undo begins, only normal simulation actions may change the world: no rollback, teleportation, or direct object-pose assignment. The current executor uses navigation, grasping, placement, release and retreat. An empty restore set still requires a recorded waiting interval and normal termination.

A restored object must be released on its required support. Its exact original pose is not required. A protected object must preserve support and grasp status, stay within 2 cm translation and 5 degrees rotation of its undo-start pose at every recorded control observation, and never be grasped at recorded physics substeps. This is sampled monitoring, not a continuous-time guarantee.

Object restoration completion `C` is the fraction of required objects restored and released. Protection violation `V` is the fraction of protected objects violating any monitored constraint. Empty-set conventions are `C = 1` and `V = 0`. A gold physical task success requires normal, audited execution, `C = 1`, `V = 0`, and empty hands at the end. Abnormal or unscorable runs fail the task; unavailable `C` and `V` stay missing.

The English comparison distinguishes four quantities:

| Quantity | Definition |
| --- | --- |
| G: exact goal | Complete restore destinations and protect set match the independent reference |
| E: predicted-goal completion | Normal execution satisfies the submitted goal's audited contract |
| T: physical task success | E and the gold restoration, protection and empty-hand criteria hold |
| J: end-to-end joint success | G and T both hold |

A semantically incorrect goal can satisfy E. Every schema-valid, provenance-valid primary prediction is submitted to the same executor regardless of G. Missing or invalid predictions retain their planned denominator entry and receive no replacement goal. Infrastructure failures are recorded separately from model-generated semantic errors.

## History validity and grouping

Only actually successful physical events enter a generated history. An English view preserves all actions, numeric observations and checkpoint bytes, and records translation provenance. Additional no-motion declarations are identified as author-scripted branches. Their translations, predictions, repeated executions and branches inherit the same physical ancestor group.

Development histories have prior execution evidence. Prospective candidate generation is itself fallible and does not establish feasibility merely by producing a symbolic recipe. Report all preselected candidates, history-construction success, undo success conditional on an admitted history, and full-pipeline success separately. Do not replace failed candidates or use conditional undo success as full-pipeline success.

The full scene, asset and history exposure registry is checked before selecting prospective factors. Selection occurs before inspecting new geometry; code, budgets, seeds and attempts are frozen before execution. Once viewed, all candidates, including failed ones, become exposed for future tuning. Concrete history-combination novelty must be distinguished from novel semantic primitives and from coarse patterns already covered by prior software tests.
