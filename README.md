<h1 align="center">M³Eval: Multi-Modal Memory Evaluation through Cognitively-Grounded Video Tasks</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2606.05008"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b" alt="Paper"></a>
  <a href="https://pku-value-lab.github.io/m3eval-homepage/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
  <a href="https://huggingface.co/datasets/PKU-VaLuE-Lab/m3eval"><img src="https://img.shields.io/badge/Hugging%20Face-Dataset-yellow" alt="Hugging Face Dataset"></a>
  <a href="https://github.com/PKU-VaLuE-Lab/m3eval"><img src="https://img.shields.io/badge/GitHub-Code-black" alt="GitHub Code"></a>
</p>

<p align="center">
  <a href="https://jie-1203.github.io/">Jie Huang</a><sup>1,*</sup>&emsp;
  <a href="https://liuruixun.github.io/">Ruixun Liu</a><sup>1,*</sup>&emsp;
  <a href="https://github.com/PKU-VaLuE-Lab/m3eval">Sirui Sun</a><sup>1</sup>&emsp;
  <a href="https://yyxxyy574.github.io/xinyiyang.github.io/">Xinyi Yang</a><sup>1</sup>&emsp;
  <a href="https://yinli-vision.github.io/">Yin Li</a><sup>2</sup>&emsp;
  <a href="https://yzhu.io/">Yixin Zhu</a><sup>1</sup>&emsp;
  <a href="https://yiwuzhong.notion.site/homepage">Yiwu Zhong</a><sup>1,†</sup>
</p>

<p align="center">
  <sup>1</sup>Peking University&emsp;
  <sup>2</sup>University of Wisconsin-Madison
</p>

<p align="center">
  * Equal contribution.&emsp;† Corresponding author.
</p>

## News

- **2026-11-16**: update codebase.
- **2026-6-4**: We released the M³Eval benchmark, code, and project page.

## M³Eval Overview

<p align="center">
  <img src="figures/readme/overview.png" alt="M³Eval overview" width="95%">
</p>

### Abstract

As multi-modal models advance towards long-form video understanding, memory emerges as a critical capability. Despite substantial effort in developing video datasets and benchmarks, existing work primarily focuses on perception and reasoning, without systematically evaluating memory: what models retain, how faithfully information is preserved, and how robust memory remains under interference.

To address this gap, we introduce M³Eval, the first comprehensive evaluation framework and benchmark for probing different memory dimensions in multi-modal models. Grounded in cognitive psychology, our design features carefully constructed tasks isolating key aspects of memory. Leveraging M³Eval, we conduct extensive experiments across representative multi-modal models, revealing consistent weaknesses and distinctive behaviors.

We find that models struggle to maintain disentangled representations when processing parallel video streams, exhibit interference patterns differing substantially from those observed in human memory, ground memory sources more reliably in the spatial domain than the temporal domain, and demonstrate limited symbolic memory. Collectively, our benchmark provides a valuable resource for future research, whereas our findings highlight memory as a fundamental yet underexplored capability and offer insights for designing more effective memory mechanisms in multi-modal models.

## Main Results

### Divided Attention

![Divided Attention main result](figures/readme/divided-attention-result.png)

Accuracy (%) on three divided attention metrics under the split-screen setting without swaps and with frequent left/right swaps.

### Memory Interference

![Memory Interference main result](figures/readme/memory-interference-result.png)

Proactive: first video (V1) interferes with recall of the second video (V2); retroactive: second video (V2) interferes with recall of the first video (V1). Delta denotes proactive minus retroactive.

### Interleaved Events

![Interleaved Events main result](figures/readme/interleaved-events-result.png)

Accuracy (%) on four interleaved reconstruction metrics.

### N-Back

![N-Back main result](figures/readme/n-back-result.png)

Average accuracy under two symbolic attributes, scene and action, averaged over all K and N configurations.

## Usage

### Install

Requires Linux x86-64, Git, curl, and a working NVIDIA GPU driver (`nvidia-smi`).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
git clone https://github.com/PKU-VaLuE-Lab/m3eval.git
cd m3eval
uv sync --locked
```

`uv sync` installs Python 3.10 and the locked dependencies. Run the remaining commands from the repository root.

### Download the Dataset

The benchmark is composed from existing public benchmarks. Users must follow the original licenses and usage terms of each source dataset.

```bash
uv run python scripts/open_source/download_dataset.py --local-dir data/m3eval
```

Downloads resume automatically and verify file checksums. Add `--include-prefix-videos` to include the original N-back concatenated videos.

Download InternVL3.5-8B:

```bash
uv run hf download OpenGVLab/InternVL3_5-8B \
  --revision 9bb6a56ad9cc69db95e2d4eeb15a52bbcac4ef79 \
  --local-dir models/InternVL3_5-8B
```

### Evaluation

```bash
bash lmms-eval/scripts/run_m3eval_internvl.sh \
  --data-root data/m3eval \
  --model-path models/InternVL3_5-8B \
  --task m3eval --gpus 0,1 \
  --output-root lmms-eval/output/m3eval/full
```

For a quick smoke run, add `--limit-per-task 1 --nback-tests-per-split 1` and use a different output directory. Use `--gpus 0,1,2,3,4,5,6,7` to run with eight GPUs.

Useful task names:

- `m3eval`
- `m3eval_memory_interference`
- `m3eval_split_screen`
- `m3eval_interleaved`
- `m3eval_nback`

The split-screen and interleaved tasks include source-memory and false-memory questions. The runner saves model responses, scores, and `m3eval_paper_tables.md` in the output directory for comparison with the paper. Use a new output directory for each run; uncovered metrics in partial runs appear as `N/A`.

## Dataset Examples

### Divided Attention

![Divided Attention dataset example](figures/readme/divided-attention-example.png)

Simultaneous memory for two side-by-side videos.

<details>
<summary>Click to expand more examples</summary>

### Memory Interference

![Memory Interference dataset example](figures/readme/memory-interference-example.png)

Interference between sequentially presented videos.

### Interleaved Events

![Interleaved Events dataset example](figures/readme/interleaved-events-example.png)

Memory reconstruction from temporally interleaved clips.

### N-Back

![N-Back dataset example](figures/readme/n-back-example.png)

Decide whether the final clip matches the clip N positions earlier.

</details>

## Citation

If you use M³Eval in your work, please cite:

```bibtex
@article{huang2026m3eval,
  title   = {M3Eval: Multi-Modal Memory Evaluation through Cognitively-Grounded Video Tasks},
  author  = {Huang, Jie and Liu, Ruixun and Sun, Sirui and Yang, Xinyi and Li, Yin and Zhu, Yixin and Zhong, Yiwu},
  journal = {arXiv preprint arXiv:2606.05008},
  year    = {2026}
}
```
