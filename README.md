# NeuroNCA

Phase 0 research codebase: **Local Importance Signal Validation**.

Before any growth/pruning orchestrator work, Phase 0 asks whether a locally computable quantity (local activity + a globally broadcast residual) can rank neurons and connections well enough to stand in for gradient- and ablation-based ground truth — on a stationary MNIST task and after a task shift.

The reasoning network trains with standard backpropagation only. Local signals are **observed alongside** training and never update weights or change topology.

## Install

```bash
pip install -e ".[dev]"
```

Requires Python ≥ 3.10.

## Status

Scaffold only: installable package layout, frozen default YAML schema, and design docs. Training, signals, and evaluation land in later PRs. Full single-command reproduction is not yet available.

## Design

See [`docs/design/phase0_local_importance_signal_validation.md`](docs/design/phase0_local_importance_signal_validation.md) for the full Phase 0 specification (signal formulas, GT measures, thresholds, multi-seed verdict, artifact layout).

## Layout

```
configs/          # default.yaml, stationary.yaml, task_shift.yaml
src/neuronca/     # package (models, data, training, broadcast, signals, …)
tests/
results/          # experiment outputs (gitignored except .gitkeep)
docs/design/      # design documents
```

## License

Research prototype; no license asserted yet.
