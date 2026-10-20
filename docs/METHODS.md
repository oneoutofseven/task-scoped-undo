# Goal inference

The portable entry point is `task-undo-predict`. It accepts only the public
narrative schema shown in [`examples/book_adopted.json`](../examples/book_adopted.json):
object and support catalogs, current supports, an ordered activity log and a
cancellation request. No reference answers are supplied.

| Method flag | Calls | Selected output |
| --- | ---: | --- |
| `direct` | 2 | `goal.json` |
| `facts-rule` | 2 | `rule_goal.json` |
| `no-history` | 2 | `goal.json` |

Direct prediction produces a goal and performs one fixed review without gold
feedback. The no-history control uses the same procedure after removing history.

Facts + rule first extracts ordered placement/use events with text references.
The second model call identifies the cancelled task and emits a model goal.
The deterministic rule separately reads the extracted events and selected task.
It does not correct factual errors using the reference answer.

Both readouts are saved for the shared pipeline: `goal.json` is the model
readout; `rule_goal.json` is the facts + rule readout. An invalid model readout
does not suppress a valid rule readout, or vice versa. The command exit status
reflects the selected method.

## Outputs

Each new output directory contains the public input, model configuration,
source snapshot, `first/` and `second/` request/response records, and `status.json`.
Model errors and invalid outputs remain recorded. There are no retries based
on semantic correctness. A saved goal establishes neither its correctness nor
physical completion.

`ChatModel` calls the configured `/chat/completions` endpoint. Table output
uses JSON-schema response constraints. Configure a server that supports this
interface and the selected model. The client bypasses shell proxy settings;
credentials are taken from `api_key_env` and are not written into run records.

The supplied endpoint configuration is a local example. The repository does
not launch, stop or download a model service automatically.

## Connecting predictions to physics

`task-undo execute-language` supports saved readouts including
`object_table_direct_v1` and `object_table_rule_v1`. Before dispatch, it replays
the saved responses and checks the request, object/support catalogs and current
supports against the physical episode.

A benchmark must additionally bind its narrative to the actual physical
history and checkpoint through a frozen protocol. The small JSON examples are
synthetic and have no physical checkpoint. Do not attach them to an unrelated
scene to manufacture an end-to-end result.

## Other code included

`agents/comparators/` contains goal-level ReAct and CaP adapters. They are not
full reproductions of those papers' robot systems. Native CaP requires Linux,
CPython 3.10 and bubblewrap at the paths specified in `native_program.py`;
it is separate from the CPU quick start and fails if isolation is unavailable.
