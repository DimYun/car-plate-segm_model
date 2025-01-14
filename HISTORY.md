# History of experiments

ClearML is closed by default, except for best experiment logs (see [README](README.md))


## 30.04.2024, exp-1, Dmitrii

- https://app.clear.ml/projects/8a4a72ee644148f781e5ba6beaaf8c65/experiments/55f387787ef14553ba0cce62feab42b6/artifacts/input-model/13794462b4884fc69123e549f493ffcb?columns=selected&columns=type&columns=name&columns=tags&columns=status&columns=project.name&columns=users&columns=started&columns=last_update&columns=last_iteration&columns=parent.name&order=-last_update&filter=
- segmentation_models_pytorch.FPN + resnet50

### Good
- 

### Bad
- Total metric IoU 0.822, forget to turn off model early stop, seems to oscillate across global extreme with AdamW with
  noise reduction and early stop disable train too early.

### Ideas
- Disable early stop

---


## 01.05.2024, exp-2, Dmitrii

- https://app.clear.ml/projects/8a4a72ee644148f781e5ba6beaaf8c65/experiments/36cd7b5e58ad490ca74676ffd11577ec/output/execution
- segmentation_models_pytorch.FPN + resnet50

### Good
- Total metric IoU 0.857, not bad for test project

### Bad
- 

### Ideas
- Train seems a little strange with periodically noises, need to check data and clean them a little.
- Try another optimizer.

---
