# Pi0.5 Proprioception

This directory mirrors the Isaac-GR00T `proprioception` layout for Pi0.5
observation experiments.

Current scope:
- `run_vlm_question_attention.py`
  - Run prompt-conditioned VLM attention on offline PlacePhone frames
  - Save per-view overlays, `.npy` heatmaps, and a `summary.json`

Notes:
- This repository's `realworld_deploy/` code is complete, but the low-level
  `openpi.models*` sources are not checked into this worktree.
- The script therefore uses a detected OpenPI runtime checkout as backend.
  By default it tries:
  1. `$OPENPI_RUNTIME_ROOT`
  2. `../openpi`
  3. `/data3/yinmenghao/code/openpi`

Example:

```bash
export CUDA_VISIBLE_DEVICES=3
/home/jikangye/workspace/baselines/vla-baselines/openpi/.venv/bin/python \
  proprioception/run_vlm_question_attention.py \
  --episode-id 155 \
  --step 400 \
  --layers 12
```
