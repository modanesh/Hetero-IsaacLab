![Isaac Lab](docs/source/_static/hetero_isaaclab.gif)

---

# Hetero-IsaacLab (Fork of Isaac Lab)

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
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

### Training Results and Insights
For a comprehensive breakdown of these experimental results, including the detailed methodology and the complete set of performance plots, the full report is available in [this WandB report](https://wandb.ai/modanesh/Hetero-Isaac/reports/Hetero-IsaacLab-Experiments--VmlldzoxNjQ4NTMzNw).

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

## Isaac Sim Version Dependency

Isaac Lab is built on top of Isaac Sim and requires specific versions of Isaac Sim that are compatible with each release of Isaac Lab. Below, we outline the recent Isaac Lab releases and GitHub branches and their corresponding dependency versions for Isaac Sim.

| Isaac Lab Version             | Isaac Sim Version         |
| ----------------------------- | ------------------------- |
| `main` branch                 | Isaac Sim 4.5 / 5.0 / 5.1 |
| `v2.3.X`                      | Isaac Sim 4.5 / 5.0 / 5.1 |
| `v2.2.X`                      | Isaac Sim 4.5 / 5.0       |
| `v2.1.X`                      | Isaac Sim 4.5             |
| `v2.0.X`                      | Isaac Sim 4.5             |

## Contributing to Isaac Lab

We wholeheartedly welcome contributions from the community to make this framework mature and useful for everyone.
These may happen as bug reports, feature requests, or code contributions. For details, please check our
[contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html).

## Troubleshooting & Support

* Please see the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section for common fixes or [submit an issue](https://github.com/isaac-sim/IsaacLab/issues).
* For issues related to Isaac Sim, we recommend checking its [documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html) or opening a question on its [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).
* Please use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for discussing ideas, asking questions, and requests for new features.

## License

The Isaac Lab framework is released under [BSD-3 License](LICENSE). The `isaaclab_mimic` extension and its corresponding standalone scripts are released under [Apache 2.0](LICENSE-mimic). The license files of its dependencies and assets are present in the [`docs/licenses`](docs/licenses) directory.

## Acknowledgement & Citation

If you use Isaac Lab in your research, please cite their technical report:

```bibtex
@article{mittal2025isaaclab,
  title={Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning},
  author={Mayank Mittal and Pascal Roth and James Tigue and Antoine Richard and Octi Zhang and Peter Du and Antonio Serrano-Muñoz and Xinjie Yao and René Zurbrügg and Nikita Rudin and Lukasz Wawrzyniak and Milad Rakhsha and Alain Denzler and Eric Heiden and Ales Borovicka and Ossama Ahmed and Iretiayo Akinola and Abrar Anwar and Mark T. Carlson and Ji Yuan Feng and Animesh Garg and Renato Gasoto and Lionel Gulich and Yijie Guo and M. Gussert and Alex Hansen and Mihir Kulkarni and Chenran Li and Wei Liu and Viktor Makoviychuk and Grzegorz Malczyk and Hammad Mazhar and Masoud Moghani and Adithyavairavan Murali and Michael Noseworthy and Alexander Poddubny and Nathan Ratliff and Welf Rehberg and Clemens Schwarke and Ritvik Singh and James Latham Smith and Bingjie Tang and Ruchik Thaker and Matthew Trepte and Karl Van Wyk and Fangzhou Yu and Alex Millane and Vikram Ramasamy and Remo Steiner and Sangeeta Subramanian and Clemens Volk and CY Chen and Neel Jawale and Ashwin Varghese Kuruttukulam and Michael A. Lin and Ajay Mandlekar and Karsten Patzwaldt and John Welsh and Huihua Zhao and Fatima Anes and Jean-Francois Lafleche and Nicolas Moënne-Loccoz and Soowan Park and Rob Stepinski and Dirk Van Gelder and Chris Amevor and Jan Carius and Jumyung Chang and Anka He Chen and Pablo de Heras Ciechomski and Gilles Daviet and Mohammad Mohajerani and Julia von Muralt and Viktor Reutskyy and Michael Sauter and Simon Schirm and Eric L. Shi and Pierre Terdiman and Kenny Vilella and Tobias Widmer and Gordon Yeoman and Tiffany Chen and Sergey Grizan and Cathy Li and Lotus Li and Connor Smith and Rafael Wiltz and Kostas Alexis and Yan Chang and David Chu and Linxi "Jim" Fan and Farbod Farshidian and Ankur Handa and Spencer Huang and Marco Hutter and Yashraj Narang and Soha Pouya and Shiwei Sheng and Yuke Zhu and Miles Macklin and Adam Moravanszky and Philipp Reist and Yunrong Guo and David Hoeller and Gavriel State},
  journal={arXiv preprint arXiv:2511.04831},
  year={2025},
  url={https://arxiv.org/abs/2511.04831}
}
```

In addition to Isaac Lab's acknowledgements and citations, please consider citing the Hetero-Isaac repository and technical blog post if you use it in your research:

```bibtex
@software{heteroisaac,
  author  = {Danesh, Mohamad H.},
  title   = {Hetero-Isaac: Heterogeneous Quadrupedal Simulation built atop Isaac Lab},
  year    = {2026},
  url     = {https://github.com/modanesh/Hetero-IsaacLab},
  license = {BSD-3-Clause},
  doi     = {10.5281/zenodo.19488668}
}

@misc{danesh2026heterogeneous,
  title        = {Heterogeneous Environments in Isaac Lab},
  author       = {Danesh, Mohamad H.},
  year         = {2026},
  howpublished = {Technical Blog Post},
  url          = {https://modanesh.github.io/blog/hetero-isaaclab},
  doi          = {10.17605/OSF.IO/M4DGU}
}
```
