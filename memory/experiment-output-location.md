---
name: experiment-output-location
description: Future experiment outputs must go to outputs/ not experiments/
metadata:
  type: project
---

All future experiment runs should use `--output_dir outputs/exp_XXX_name` (NOT `experiments/`).

The teammate has reconfigured `.gitignore` so that `outputs/` tracks experiment logs (config.yaml, results.json, training_history.csv, training_log.jsonl) while ignoring model weights (best_model.pth, *.pth, checkpoints/).

**Why:** Teammate set this up as the project standard. The `outputs/` directory has cleaner gitignore rules already in place.

**How to apply:** When running training, always use:
```bash
python scripts/train.py --config ... --output_dir outputs/exp_XXX_name ...
```
