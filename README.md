# Quantum-Topological Intelligence (QTI)

QTI is an experimental research project exploring whether topological representations, persistent-homology features and structured memory dynamics can provide measurable benefits to learning systems.

The project name is historical. The current research framing is topology-inspired machine learning, not a claim to quantum computation, consciousness or a replacement for neural networks.

## Research question
Can topological descriptors of data, memory or model state improve prediction, robustness, interpretability or adaptation compared with well-chosen conventional baselines?

## Working hypotheses
- Persistent-homology features may capture structural information that ordinary feature pipelines miss in selected tasks.
- Topology-aware memory updates may provide useful state representations for certain sequential or dynamical problems.
- Any claimed benefit must survive ablation, baseline comparison and reproducible evaluation.

These are hypotheses, not established findings.

## Repository contents
- QTI_Core/ — experimental sensor, memory, actor, phase and visualization modules.
- tests/ — unit tests for current components.
- demo_qti_core.py and demo_weight_sensor.py — exploratory demonstrations.
- QTI_Technical_Details.md and QTI_Publication_Draft.md — design and writing drafts.
- DEV_PLAN.md and CHANGELOG.md — development notes.

## Evaluation standard
A meaningful experiment should specify the task and dataset, leakage controls, conventional baselines, metrics, ablations, random seeds, compute budget and failure cases. Report negative results. Do not interpret phase changes as awareness or equate mathematical topology with subjective experience.

## Status
Research prototype. Test coverage does not establish scientific validity or practical advantage. No production readiness is claimed.

## Development
Install dependencies from requirements.txt and run pytest. Check the actual toolchain and current test status before relying on results.

## License
MIT — see LICENSE.

## Connections to other projects

QTI investigates representation and memory structure. It sits beside [TORUS](https://github.com/Personaz1/TORUS), which explores topology-inspired generative processes, but the two should not be assumed to share a mechanism or result.

Potential research links:
- [decentralized-ai-child](https://github.com/Personaz1/decentralized-ai-child): test whether structural descriptors or memory dynamics help multi-model coordination.
- [TORI](https://github.com/Personaz1/TORI): contribute precise representation and evaluation questions to the broader AGI architecture investigation.
- [LongGapsBetweenPrimes](https://github.com/Personaz1/LongGapsBetweenPrimes) and [math-challenge-millennium](https://github.com/Personaz1/math-challenge-millennium): use formal specifications where a mathematical claim can be stated precisely.

Keep three layers distinct: mathematical definitions, implemented computational mechanisms, and interpretations about intelligence or cognition. A resemblance between diagrams or vocabulary is not evidence of a shared underlying process.
