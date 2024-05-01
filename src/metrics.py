from torchmetrics import MetricCollection, JaccardIndex, Dice


def get_metrics() -> MetricCollection:
    return MetricCollection({
        'iou': JaccardIndex(
            num_classes=1, task='binary'
        )
    })
