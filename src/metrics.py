"""Module for setting metrics for validation and test"""

from torchmetrics import JaccardIndex, MetricCollection


def get_metrics() -> MetricCollection:
    """
    Set metrics structure for model
    :return: MetricCollection
    """
    return MetricCollection(
        {
            "iou": JaccardIndex(
                num_classes=1,
                task="binary",
            ),
        },
    )
