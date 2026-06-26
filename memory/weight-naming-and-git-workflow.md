---
name: weight-naming-and-git-workflow
description: Weight file naming convention and git workflow rules
metadata:
  type: project
---

All model weight files must be named as `official_0xx_best_model.pth` (with experiment ID prefix), placed inside `outputs/official_0xx/`.

Example:
```
outputs/official_014/official_014_best_model.pth
outputs/official_017_avg/official_017_avg_best_model.pth
```

**Why:** Avoids confusion between experiments — plain `best_model.pth` is not distinguishable.

**Git workflow:** From now on, commit and push directly to `main`. No more feature branches. This simplifies collaboration since the teammate also pushes directly to main.

**How to apply:**
```bash
cd d:/WEATHER_AI/weather-classification
git checkout main
git pull
# ... run experiments ...
cp outputs/official_0xx/best_model.pth outputs/official_0xx/official_0xx_best_model.pth
git add outputs/official_0xx/
git commit -m "feat: ..."
git push origin main
```
