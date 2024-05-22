from typing import Literal  # Для типизации строковых аргументо
import torch
from torch import Tensor, nn
EPSILON = 1e-6
NormalizationOptionT = Literal["sigmoid", "softmax", "none"]  # Опция нормализации (sigmoid, softmax, none)

class DiceLoss(nn.Module):
    def __init__(
        self,
        normalization: NormalizationOptionT = "sigmoid",
        weights: Tensor | None = None,
        epsilon: float = EPSILON,
    ) -> None:
        super().__init__()

        if normalization == "sigmoid":
            self.normalization: nn.Module = nn.Sigmoid()
        elif normalization == "softmax":
            self.normalization = nn.Softmax(dim=1)
        elif normalization == "none":
            self.normalization = nn.Identity()
        else:
            raise ValueError

        self.register_buffer("weights", weights)
        # Используется для сохранения тензора весов в модели,
        # чтобы он был частью состояния модели, но не обновлялся градиентами

        self.epsilon = epsilon

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        probs = self.normalization(inputs)
        per_channel_dice = compute_dice_per_channel(
            probs=probs, targets=targets, epsilon=self.epsilon, weights=self.weights
        )
        return 1.0 - torch.mean(per_channel_dice)


# Вычисление коэффициента Dice для каждого канала
def compute_dice_per_channel(
    probs: Tensor,
    targets: Tensor,
    epsilon: float = EPSILON,
    weights: Tensor | None = None,
) -> Tensor:

    assert probs.size() == targets.size()

    probs = probs.transpose(1, 0).flatten(2)
    targets = targets.transpose(1, 0).flatten(2).float()

    numerator = (probs * targets).sum(-1)
    if weights is not None:
        numerator = weights * numerator

    denominator = (probs + targets).sum(-1)

    return torch.mean(2 * (numerator / denominator.clamp(min=epsilon)), dim=1)