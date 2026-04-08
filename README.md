![Isaac Lab](docs/source/_static/hetero_isaaclab.gif)

---

# Hetero-IsaacLab (Fork of Isaac Lab)

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)

**Notice:** This repository is a fork of the main Isaac Lab repo at commit [752be19b](https://github.com/isaac-sim/IsaacLab/tree/752be19bade88c1f1e2a06a1ca6519baafbba216). After that point, all subsequent modifications for heterogeneous training have been consolidated into a single commit to provide a clean, streamlined history.

**Notice:** This project is different from [IsaacLab-HARL](https://github.com/DIRECTLab/IsaacLab-HARL) which focuses on heterogeneous _multi-agent_ learning in Isaac Lab. In this project, we focus on heterogeneous _single-robot_ learning, where each environment contains a single robot but the robots across environments are different. This allows us to train morphology-agnostic policies across multiple quadrupedal robots simultaneously.

**Isaac Lab** is a GPU-accelerated, open-source framework designed to unify and simplify robotics research workflows, such as reinforcement learning, imitation learning, and motion planning. Built on NVIDIA Isaac Sim, it combines fast and accurate physics and sensor simulation, making it an ideal choice for sim-to-real transfer in robotics.

## 🤖 Heterogeneous Multi-Robot Training

This fork introduces **Hetero-IsaacLab**, a specialized architecture for training morphology-agnostic locomotion policies across multiple heterogeneous **quadrupedal** robotic environments simultaneously. 

Most physics simulators and RL frameworks assume homogeneity, making it difficult to train universal controllers. This repository bridges that gap, providing concrete advantages:
* **Morphology-Agnostic Feature Learning:** The policy is forced to learn fundamental locomotion principles that transcend specific hardware rather than memorizing robot-specific quirks.
* **Efficient Multi-Platform Deployment:** Training 8 robot types heterogeneously uses the same compute as training 1 robot type, eliminating the need to maintain separate codebases and models.
* **Better Exploration:** Different morphologies explore different regions of the state-action space naturally (e.g., lighter robots discover high-speed gaits, heavier ones excel at stability).

### Key Architecture Features
* **Heterogeneous Configuration System:** Custom config classes with dynamic environment assignment and reward filtering.
* **Observation & Action Unification:** Enforces an "ANYmal Joint-Major" format, mapping diverse joint orders (e.g., Spot, Unitree) to a standard policy format.
* **Index Mapping System:** Efficient conversion between global environment IDs and robot-local indices.
* **Comprehensive Domain Randomization:** Handles extreme morphological quirks with flexible reset randomizations (mass, CoM, friction) and interval randomization for external disturbances.

## Getting Started

Our [documentation page](https://modanesh.github.io/blog/hetero-isaaclab) provides everything you need to get started with this framework.

### Installation
Installation is quite similar to the original Isaac Lab's default installation, cloning this repository instead of the original one.

```bash
# Clone Hetero-IsaacLab
git clone https://github.com/modanesh/Hetero-IsaacLab.git
cd Hetero-IsaacLab

# Install dependencies (and training modules, eg rsl_rl)
./isaaclab.sh --install rsl_rl
````

### Basic Training

To train on a specific subset of robots, you can pass the list with the `--quadrupeds` flag, from the list of available quadrupeds: `anymal_d,anymal_c,anymal_b,unitree_a1,unitree_go1,unitree_go2,unitree_b2,spot`.

```bash
# Train on all 8 robots with 4096 environments
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Velocity-Flat-HeteroQuadruped-v0 \
    --quadrupeds anymal_d,anymal_c,anymal_b,unitree_a1,unitree_go1,unitree_go2,unitree_b2,spot
```

## Acknowledgement & Citation

In addition to Isaac Lab's acknowledgements and citations, please consider citing the Hetero-Isaac repository and technical blog post if you use it in your research:

```bibtex
@misc{heteroisaac,
  author = {Danesh, Mohamad H.},
  title     = {Hetero-Isaac: Heterogeneous Quadrupedal Simulation built atop Isaac Lab},
  year      = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {https://github.com/modanesh/Hetero-IsaacLab},
}

@misc{danesh2026heterogeneous,
  title = {Training Morphology-Agnostic Locomotion Policies with Heterogeneous Robotic Environments in {Isaac Lab}},
  author = {Danesh, Mohamad H.},
  year = {2026},
  howpublished = {Technical Blog Post},
  url = {https://modanesh.github.io/blog/hetero-isaaclab},
}
```
