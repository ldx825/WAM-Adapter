<div align="center">

<h1>WAM-Adapter</h1>
<h3>Efficiently Adapting Video-DiT World Representations to Robot Actions</h3>

<p>
  <img src="https://img.shields.io/badge/arXiv-coming%20soon-B31B1B" alt="arXiv coming soon">
  <a href="https://ldx825.github.io/WAM-Adapter-Homepage/"><img src="https://img.shields.io/badge/Project-Page-4C8BF5" alt="Project Page"></a>
  <a href="https://huggingface.co/ldx825/WAM-Adapter"><img src="https://img.shields.io/badge/Hugging%20Face-Models-FFD21E" alt="Hugging Face"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-3DA639" alt="License"></a>
</p>

<p>
  <strong>Dongxu Liu</strong><sup>1,2,*</sup>&emsp;
  <strong>Zhe Li</strong><sup>1,*</sup>&emsp;
  <strong>Zundong Ke</strong><sup>1</sup>&emsp;
  <strong>Zhipeng Zhang</strong><sup>1,†</sup>
</p>

<p>
  <sup>1</sup>AutoLab, School of Artificial Intelligence, Shanghai Jiao Tong University<br>
  <sup>2</sup>Awaken Robotics<br>
  <sup>*</sup>Equal contribution&emsp;&emsp;<sup>†</sup>Corresponding author
</p>

<img src="assets/teaser.png" alt="WAM-Adapter teaser" width="95%">

</div>

## Overview

WAM-Adapter adapts a frozen **Wan2.1-T2V-1.3B** video backbone to robot control with
**121M trainable parameters**:

- LoRA adapts the backbone with future-video supervision.
- A single forward pass processes the current observation and noisy future slots.
- Temporal-group cross-layer readout fuses Current / Early-Future / Late-Future
  representations and directly predicts an action chunk.

<p align="center">
  <img src="assets/architecture.png" alt="WAM-Adapter architecture" width="95%">
</p>

Core implementation: [WAMAdapter](src/wam_adapter/models/wan22/wam_adapter.py) and
[temporal-group readout](src/wam_adapter/models/wan22/wam_adapter_readout.py).

## Results

| Benchmark | Success rate |
|---|---|
| RoboTwin 2.0 (50 tasks, clean/randomized mean) | **89.0%** |
| LIBERO (four suites) | **98.5%** |
| LIBERO-Plus | **75.17%** |

Under the same setting, training throughput is **9.66×** that of Fast-WAM with
**98% fewer trainable parameters** and **24.52 GiB** peak training memory.
On an RTX 4090, inference takes **92.05 ms** per query with **3.41 GiB** peak memory.

## Repository Layout

```text
WAM-Adapter/
├── src/wam_adapter/     # Model, readout, datasets and training code
├── configs/             # Model, data, task and evaluation configurations
├── scripts/             # Precompute, training and evaluation entry points
├── experiments/         # RoboTwin and real-robot integration
├── tests/               # Unit tests
└── LICENSES/            # Third-party license texts
```

## Setup

Python ≥ 3.10 and a CUDA environment are required.

```bash
git clone https://github.com/ldx825/WAM-Adapter.git
cd WAM-Adapter
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu128
```

Backbone: [Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B).
Released RoboTwin and LIBERO checkpoints:
[ldx825/WAM-Adapter](https://huggingface.co/ldx825/WAM-Adapter).

## Training and Evaluation

Set the local dataset, model and cache paths in the launch scripts or through
their environment variables, then use the RoboTwin entry points:

```bash
bash scripts/precompute_wam_adapter_robotwin_full.sh
bash scripts/train_wam_adapter_robotwin_full_500k.sh
CKPT=/path/to/checkpoint.pt bash scripts/eval_robotwin.sh
```

The training launcher defaults to 8 GPUs, global batch 128 and a 500k-step schedule;
the released RoboTwin checkpoint is from step 240k. Evaluation requires a local
[RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) installation.
Real-robot data preparation and deployment scripts are in
[`experiments/real_robot/`](experiments/real_robot/).

## Citation

The arXiv release is coming soon. For now, please cite the repository:

```bibtex
@misc{wamadapter2026,
  title   = {WAM-Adapter: Efficiently Adapting Video-DiT World Representations to Robot Actions},
  author  = {Liu, Dongxu and Li, Zhe and Ke, Zundong and Zhang, Zhipeng},
  year    = {2026},
  url     = {https://github.com/ldx825/WAM-Adapter},
  note    = {Code repository},
}
```

## Acknowledgements

This codebase builds upon [Light-WAM](https://github.com/L1ziang/Light-WAM) and
[Fast-WAM](https://github.com/yuantianyuan01/FastWAM). We thank their authors for
the model, training and data infrastructure. We also thank
[Wan](https://github.com/Wan-Video/Wan2.1),
[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) and
[RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin).

## License

The project uses the [MIT License](LICENSE), retaining the FastWAM and Light-WAM
copyright notices. Included third-party components retain their original licenses
and file-level notices:

| Component | License |
|---|---|
| Wan / DiffSynth model components and loading helpers | [Apache-2.0](LICENSES/Apache-2.0.txt) |
| [LeRobot](https://github.com/huggingface/lerobot) dataset utilities | [Apache-2.0](LICENSES/Apache-2.0.txt) |
| NVIDIA utilities in `src/wam_adapter/utils/misc.py` | [Apache-2.0](LICENSES/Apache-2.0.txt) |
| [PyTorch3D](https://github.com/facebookresearch/pytorch3d) rotation utilities | [BSD-3-Clause](LICENSES/PyTorch3D-BSD.txt) |
