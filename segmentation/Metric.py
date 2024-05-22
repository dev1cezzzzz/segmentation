import torch
from torch import Tensor, nn
EPSILON = 1e-6

class MeanIoU(nn.Module):
    def __init__(
        self,
        threshold: float = 0.5,  # Порог для бинаризации предсказаний
        epsilon: float = EPSILON,  # Малое значение для предотвращения деления на ноль
        binarize: bool = True,  # Нужно ли бинаризовать предсказания
    ) -> None:
        super().__init__()

        self.threshold = threshold
        self.epsilon = epsilon
        self.binarize = binarize

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        inputs is `N x C x Spatial`,
        targets is `N x C x Spatial` or `N x Spatial`.
        """
        classes_num = inputs.shape[1]
        assert inputs.dim() == targets.dim()

        if self.binarize:
            preds = binarize_probs(
                inputs=torch.sigmoid(inputs),
                classes_num=classes_num,
                threshold=self.threshold,
            )
        else:
            preds = torch.sigmoid(inputs)

        return torch.mean(
            compute_miou_per_channel(probs=preds, targets=targets, epsilon=self.epsilon)
        )

# Вычисление Mean Intersection over Union (mIoU) для каждого канала
def compute_miou_per_channel(
    probs: Tensor,  # Тензор предсказанных вероятностей размерности `N x C x Spatial`
    targets: Tensor,  # Тензор целевых значений размерности `N x C x Spatial`
    epsilon: float = EPSILON,
    weights: Tensor | None = None,  # Тензор весов для каждого канала размерности `C x 1`
) -> Tensor:

    assert probs.size() == targets.size()

    probs = probs.transpose(1, 0).flatten(2)
    targets = targets.transpose(1, 0).flatten(2).byte()

    numerator = (probs & targets).sum(-1).float()
    if weights is not None:
        numerator = weights * numerator

    denominator = (probs | targets).sum(-1).float()

    return torch.mean(numerator / denominator.clamp(min=epsilon), dim=1)

# Бинаризация предсказанных вероятностей
def binarize_probs(inputs: Tensor, classes_num: int, threshold: float = 0.5) -> Tensor:
    # input is `N x C x Spatial`

    if classes_num == 1:
        return (inputs > threshold).byte()

    return torch.zeros_like(inputs, dtype=torch.uint8).scatter_(
        1, torch.argmax(inputs, dim=0), 1
    )
