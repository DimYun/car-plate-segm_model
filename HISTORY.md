# История экспериментов

По умолчанию ClearML закрыт, откра лучшая модель (см. [README](README.md))

## 30.04.2024, exp-1 Дмитрий

- https://app.clear.ml/projects/8a4a72ee644148f781e5ba6beaaf8c65/experiments/55f387787ef14553ba0cce62feab42b6/artifacts/input-model/13794462b4884fc69123e549f493ffcb?columns=selected&columns=type&columns=name&columns=tags&columns=status&columns=project.name&columns=users&columns=started&columns=last_update&columns=last_iteration&columns=parent.name&order=-last_update&filter=
- segmentation_models_pytorch.FPN + resnet50

### Что не зашло
- общая метрика IoU 0.822, забыл отключить остановку модели, видимо при AdamW 
  происходит приближение/удаление от локального минимума с постепенным 
  уменьшением шумов - тут остановка сработала рано

### Идеи на будущее
- отключить остановку модели

---

## 01.05.2024, exp-2 Дмитрий

- https://app.clear.ml/projects/8a4a72ee644148f781e5ba6beaaf8c65/experiments/36cd7b5e58ad490ca74676ffd11577ec/output/execution
- segmentation_models_pytorch.FPN + resnet50

### Что зашло
 
- общая метрика IoU 0.857

### Идеи на будущее

- Немного странное поведение модели с периодическими шумами. Нужно проверить 
  данные и попробовать подобрать оптимайзер.

---
