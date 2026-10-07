---
title: QFF-3D
emoji: 🧊
colorFrom: gray
colorTo: pink
sdk: docker
app_port: 7860
license: mit
pinned: false
short_description: Quantum Free-Form 3D Topology Optimisation (QUBO updates)
---

# QFF-3D: Quantum Free-Form 3D Topology Optimisation

Freeform 3D topology optimisation of STL design domains with a
quantum-annealing-compatible QUBO design update, benchmarked against MMA.
Built on FreeTO (Ibhadode, Fu & Qureshi, 2024, MIT licence).

This Space is the shared online demo behind <https://nitipong.com/qff3d>.
It runs one job at a time on a small CPU, caps meshes at MeshControl 50 and
300 iterations (the paper's examples fit), stops a job after 20 minutes and
deletes runs after 2 hours. Multi-run studies are only in the local version.

Source code, paper reproduction and the full local version:
<https://github.com/cnnitipong/Quantum-Free-Form-3D-topology>

This README is generated from `deploy/huggingface/SPACE_README.md` in that
repository; edit it there, not here.
