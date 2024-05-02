from torchmetrics import JaccardIndex, MetricCollection


def get_metrics() -> MetricCollection:
    return MetricCollection(
        {
            "iou": JaccardIndex(
                num_classes=1,
                task="binary",
            ),
        },
    )
