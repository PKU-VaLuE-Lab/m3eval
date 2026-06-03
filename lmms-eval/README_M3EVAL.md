# M³Eval in `lmms-eval`

This repository keeps the full `lmms-eval` framework and adds a public task family for the M³Eval benchmark.

## Install

```bash
uv pip install -e ".[all]"
```

## Download the Dataset

The benchmark is composed from existing public benchmarks. Users must follow the original licenses and usage terms of each source dataset.

Download the dataset and unpack it into `data/m3eval/`:

```bash
huggingface-cli download PKU-VaLuE-Lab/m3eval \
  --repo-type dataset \
  --local-dir data/m3eval

bash data/m3eval/unpack_archives.sh
```

After unpacking, `data/m3eval/` should contain:

```text
data/m3eval/
├── qa_root/
├── questions/
├── nback/
├── viewer/
└── videos/
    ├── interleaved/
    ├── memory_interference/
    ├── split_screen/
    └── nback/
```

## Run M³Eval

```bash
bash lmms-eval/scripts/run_m3eval_vllm.sh \
  --model_path /path/to/your/model \
  --task m3eval \
  --gpus 0 \
  --batch_size 1
```

For multi-GPU data-parallel evaluation, use:

```bash
bash lmms-eval/scripts/run_m3eval_sharded_vllm.sh \
  --model_path /path/to/your/model \
  --task m3eval \
  --gpus 0,1,2,3 \
  --num_processes 4 \
  --batch_size 1
```

Quick smoke:

```bash
bash lmms-eval/scripts/run_m3eval_vllm.sh \
  --model_path /path/to/your/model \
  --task m3eval_nback \
  --gpus 0 \
  --limit 1 \
  --max_frame_num 8
```

Useful task names:

- `m3eval`
- `m3eval_memory_interference`
- `m3eval_split_screen`
- `m3eval_interleaved`
- `m3eval_nback`

The script writes outputs to `lmms-eval/output/m3eval/` by default.

## Paper Tables

Pass `*_results.json` from a full `--task m3eval` run:

```bash
python lmms-eval/scripts/aggregate_m3eval_results.py \
  lmms-eval/output/m3eval/path/to/*_results.json
```

The script writes `m3eval_paper_tables.json` and `m3eval_paper_tables.md` next to the selected result file. If you aggregate a single subtask result or a smoke run, uncovered fields will appear as `N/A`.
