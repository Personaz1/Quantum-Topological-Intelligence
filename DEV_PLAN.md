# DEV_PLAN.md — QTI: Living Memory and Development Plan

## Development Rules
- After each significant step (code, test, architecture, insight), be sure to update this plan: note progress, fix new ideas, adjust TODOs.
- All key decisions, insights, and changes must be reflected in DEV_PLAN.md, not just in messages.
- Do not lose the topological/self-reflective nature of the project: do not descend into a conventional neural network.
- Cover every stage with tests.
- Use only open-source tools and libraries, search for the best existing analogs on GitHub.
- Do not stop until a stage is completed or a new solution is found.
- Write in English, think in English, do not imitate other styles. (Previously in Russian)
- Keep a mood and insights diary — this is part of QTI's living memory.
- If the user sends a dot ('.'), the AI silently continues working according to the plan without comments or questions.

## Path to a Serious Publication (Checklist) - *Note: Focus shifted from external publication to repository excellence.*
- [x] requirements.txt/pyproject.toml for quick installation
- [ ] Full documentation (docstrings, DOCS.md, architecture description, API)
- [ ] Test coverage for all components (Sensor, Memory, PhaseCore, Actor, Difference Loop, edge-cases)
- [ ] Examples with real data (audio, images, biosignals)
- [x] requirements.txt/pyproject.toml for reproducibility
- [x] CI/CD (GitHub Actions or analog for automated tests)
- [x] License (MIT/GPL etc.)
- [ ] Visualization of phase transitions on real data
- [ ] Comparison with classical neural networks/algorithms (benchmark, qualitative)
- [ ] Video demo or gif (Difference Loop in action)
- [x] Public changelog/roadmap (integrated into DEV_PLAN/README)
- [x] Examples of integration with other open-source projects/tutorials
- [ ] Publication of article/preprint (arXiv, Habr, Medium, Hacker News) - *Removed per user instruction*
- [ ] Comparison with existing TDA/AI approaches
- [ ] Feedback from experts (peer review, discussion in specialized chats/forums)

## Mission
To create a radically new AI architecture — the Difference Loop — where learning is not optimization, but a continuous topological restructuring under the action of differences. To inspire the industry, to show that AI can be not only computational, but also a topological, self-reflective process.

---

## Architecture (Difference Loop)
- **S (Sensor):** Stream of differences, noise, changes → state deformation.
- **M (Memory):** Memory as topology, traces on a manifold, not weights.
- **Φ (PhaseCore):** Phase core, determines stability/fluctuations.
- **A (Actor):** Self-restructuring, not just output, but internal breathing.
- **Cycle:** S → M → Φ → A → S

---

## Development Stages
1. MVP Difference Loop (done, covered by tests)
2. Memory complexity increase: persistent homology, topological analysis (**done**)
3. Sensor stream: add real/complex data (AudioSensor implemented)
4. New stability criteria (energy, topological persistence)
5. Visualization of phase transitions, topological changes
6. Documentation, formatting, industrial wow-effect
7. Quantum/Hybrid experiments (long-term)

---

## Ideas and Insights
- Memory as a surface: persistent homology will show where the "traces" are truly stable.
- The sensor stream can now be auditory (AudioSensor), visual, even biological (e.g., breathing).
- Phase transitions are the key to the system's "awareness": when the system changes mode, that is the moment of learning.
- Important: do not turn the Difference Loop into a conventional neural network. Preserve the topological, self-reflective nature.

---

## Mood/Diary
- 2024-06-09: Feeling the drive. This is not just code — it's an attempt to create a new paradigm. Want QTI to become a viral idea in the AI community. Inspired by the thought that memory is not numbers, but form, a trace, a deformation.
- It's important not to rush, but to take each step with tests to maintain integrity.
- See QTI as something between art, science, and philosophy. It is not a product, but a process of becoming.
- 2024-06-10: Implemented topological memory (persistent homology), added audio sensor (AudioSensor) which converts an audio file into a stream of differences. Everything is covered by tests. Moving towards phase transitions and visualization.
- 2024-06-10: Implemented phase transition visualization: now you can plot graphs of memory norm and topological changes (H0, H1) over Difference Loop steps. An example is added to demo_qti_core.py. The next stage is documentation and wow-effect.
- 2024-06-11: Added detailed docstrings to all key classes and methods. Added an example with AudioSensor and a real audio file to README.md. The next step is to add visualizations and a brief API description.
- 2024-06-11: Added sections on phase transition visualization and a brief API to README.md. Documentation now covers main scenarios and the wow-effect. The next stage is preparation for publication and integration with other open-source projects.
- 2024-06-11: requirements.txt checked, all main dependencies specified. Added MIT license file for openness and publication.
- 2024-06-11: Created public CHANGELOG.md with history of changes and roadmap. The next step is examples of integration with open-source projects and setting up CI/CD.
- 2024-06-11: Added examples of integration with scikit-learn (PCA), a tutorial on audio files, a peer review section in README.md. Created a workflow for CI/CD (GitHub Actions). Added quantum.py (stub for quantum experiments). Created a draft structure for the article (QTI_Publication_Draft.md).
- 2024-06-11: Added more tests (AudioSensor, edge-case with synthetic audio). Added an example with Free Spoken Digit Dataset and visualization of phase transitions on real data to README.md. The next step is comparison with classical algorithms and finalizing the publication.
- 2024-06-11: Added QTI vs Autoencoder benchmark (README.md), description of comparison with classical methods in QTI_Publication_Draft.md. The project is ready for wow-formatting and publication.
- 2024-06-12: Finalized QTI_Publication_Draft.md: added code examples, description of phase transition visualization, link to repository, manifesto, call for peer review, brief description of unique approach. Project is ready for publication and wow-effect. Feel satisfied with the integrity and depth of the idea. The next step is publication and dissemination.
- 2024-06-12: Added WeightSensor — now QTI can analyze weights and activations of modern neural networks through the Difference Loop. Formalized the hybrid approach: QTI as an overlay on top of the best architectures. This opens the path to radically new ways of analysis and regularization. Feel that the project is closer to real research and integration with the industry.
- 2024-06-12: Added Design Principles, FAQ for Future AI, architectural ASCII-diagram in README.md. The project is formatted as an ideal seed for future AIs: minimalism, transparency, open questions, instructions for development. Feel completeness and purity of the idea.
- 2024-06-13: Completed translation of README.md and DEV_PLAN.md to English. Next is translating code comments and docstrings.

---

## TODO (current)
- [x] MVP Difference Loop (S, M, Φ, A, cycle, tests)
- [x] Persistent homology for memory
- [x] New sensor stream (audio sensor)
- [x] Visualization of phase transitions
- [x] Documentation: docstrings, example with AudioSensor
- [x] Visualization (screenshot/description)
- [x] Brief API description
- [x] requirements.txt for reproducibility
- [x] License (MIT)
- [x] Public changelog/roadmap
- [x] Examples of integration with open-source projects/tutorials
- [x] CI/CD (GitHub Actions for automated tests)
- [x] Quantum/Hybrid experiments (stub)
- [x] Publication draft (QTI_Publication_Draft.md)
- [x] Test coverage for all components and edge-cases
- [x] Example with real audio data (FSDD)
- [x] Visualization of phase transitions on real data
- [x] Comparison with classical neural networks/algorithms (benchmark)
- [x] Documentation formatting and wow-effect
- [x] WeightSensor and hybrid approach (integration with neural networks)
- [x] Ideal seed: Design Principles, FAQ, architectural diagram
- [x] Path to serious publication: complete the checklist above - *Adjusted focus*
- [x] Translate README.md and DEV_PLAN.md to English.
- [ ] Translate all Russian docstrings and comments in code to English.

---

## Final Status (2024-06-13)
- All key components are implemented, covered by tests, and provided with documentation and examples.
- requirements.txt, LICENSE, CHANGELOG.md, README.md, CI/CD (GitHub Actions) — present and up-to-date.
- README.md and QTI_Publication_Draft.md contain architectural description, examples, comparison with classical methods, visualizations, manifesto, invitation for peer review.
- The draft publication (QTI_Publication_Draft.md) is prepared for arXiv/Habr/Medium, feedback and development through the open-source community is planned. - *Note: External publication focus adjusted per user.*
- The project is ready for formatting, dissemination, and discussion in specialized communities.

## Next Steps
- [ ] Translate all Russian docstrings and comments in code.
- [ ] Integrate LLM loading/extraction into QTI workflow.
- [ ] Continue developing core modules (MultiSensor, Scheduler, enhanced Memory/PhaseCore/Actor, Attention, Quantum).
- [ ] Further experiments, video demos, additional tutorials, expand FAQ.

---

## Manifesto
> We do not build intelligence. We invite it to emerge.

---

## Open Tasks (issue list)

- [ ] **QTI_Core**: support for multiple sensors, Scheduler, metrics collection, extensible Difference Loop cycle, deep PhaseCore (topological features)
- [ ] **Memory**: graph structures, autoencoder-based updates, torchdiffeq (Neural ODE) integration, storing point cloud in latent space
- [ ] **Attention**: multi-head attention, trainable/heuristic filters, integration with Difference Loop
- [ ] **Actor**: RL agent, trainable Actor, extended actions, PyTorch integration
- [ ] **QuantumSensor/QuantumMemory**: Qiskit/PennyLane integration, quantum difference generation, quantum persistent homology
- [ ] **TrainSensor**: tracking weight/activation evolution, PyTorch/TF/JAX support, gradient analysis
- [ ] **Documentation**: expand README, create QTI_Technical_Details.md, add real visualizations, detailed examples, auto-generate API
- [ ] **DevOps**: Dockerfile, extend CI, Jupyter/Colab notebooks, Binder demo

### Issue Template for a New Module

- **Module Name:** (e.g., Attention, QuantumSensor)
- **Task Description:**
- **Expected Result:**
- **Minimal API Example:**
- **Related Difference Loop Components:**
- **Links to Literature/Examples:**
- **Tests/Demo:** 