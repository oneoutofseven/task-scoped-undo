# Optional environment records

The CPU quick start and included tests require only Python and the local package.

| File | Purpose |
| --- | --- |
| `model.txt` | Recorded vLLM / Transformers versions for a separate model environment |
| `runtime-fixes.txt` | Recorded simulator compatibility fixes |
| `constraints-primitives.txt` | Constraints used when installing manipulation dependencies |

These are partial environment records, not complete lockfiles. Installing the
model dependencies into the simulator environment is not part of the setup.
See [SIMULATION.md](../docs/SIMULATION.md).
