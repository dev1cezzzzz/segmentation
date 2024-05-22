import torch
from torch import Tensor, nn
from torch.nn import functional as F  # Предоставляет различные функциональные API, такие как функции активации и свертки


class UNet(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, bilinear: bool = True
    ) -> None:
        super().__init__()

        self.bilinear = bilinear # Флаг, определяющий, использовать ли билинейную интерполяцию для upsampling

        self.in_conv = DoubleConv(in_channels, 64) # Обработка входного изображения

        self.down1 = _Down(64, 128)  # Слои downsampling (снижение разрешения)
        self.down2 = _Down(128, 256)
        self.down3 = _Down(256, 512)
        channels_factor = 2 if bilinear else 1
        self.down4 = _Down(512, 1024 // channels_factor)

        self.up1 = _Up(1024, 512 // channels_factor, bilinear=bilinear) # Слои upsampling (увеличение разрешения)
        self.up2 = _Up(512, 256 // channels_factor, bilinear=bilinear)
        self.up3 = _Up(256, 128 // channels_factor, bilinear=bilinear)
        self.up4 = _Up(128, 64, bilinear=bilinear)

        self.out_conv = OutConv(64, out_channels) # Финальная свертка


    def forward(self, x: Tensor) -> Tensor:
        x1 = self.in_conv(x) # Обрабатывает входное изображение через двойную свертку
        x2 = self.down1(x1) # Применяет первый уровень downsampling
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up4(self.up3(self.up2(self.up1(x5, x4), x3), x2), x1)
        # Последовательно применяет уровни upsampling, объединяя карты признаков с соответствующими уровнями downsampling
        return self.out_conv(x)

# Реализует двойную свертку с нормализацией и активацией Leaky ReLU для извлечения признаков
class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels: int, # количество входных каналов
        out_channels: int, # количество выходных каналов
        mid_channels: int | None = None, # количество каналов на промежуточном слое (если не указано, равно out_channels)
        kernel_size: int = 3, # размер ядра свертки
        padding: int = 1, # размер паддинга для свертки (количество пикселей, добавляемых к каждому краю входного изображения)
    ) -> None:
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.double_conv(x)

# Включает MaxPooling и двойную свертку
class _Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), # Уменьшает размер изображения в два раза, сохраняя количество каналов
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.maxpool_conv(x)


class _Up(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = True, # операция билинейного интерполирования - увеличить размер входного тензора x
            # в два раза ( Это позволяет получить промежуточное представление, близкое к оригинальному размеру)
        kernel_size_transp: int = 2,
        stride_transp: int = 2,
    ) -> None:
        super().__init__()

        if bilinear:
            self.up: nn.Upsample | nn.ConvTranspose2d = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )
            self.conv = DoubleConv(
                in_channels, out_channels, mid_channels=in_channels // 2
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels,
                in_channels // 2,
                kernel_size=kernel_size_transp,
                stride=stride_transp,
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: Tensor, x_skip: Tensor) -> Tensor:
        x = self.up(x)

        # input's shape is [N x C x H x W]
        diff_h = x_skip.size()[2] - x.size()[2]
        diff_w = x_skip.size()[3] - x.size()[3]

        x = F.pad(  # pylint: disable=not-callable
            x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2]
        ) # подгоняем размер x до размера x_skip

        x = torch.cat([x_skip, x], dim=1)
        return self.conv(x)

# Окончательная свертка для получения выходного изображения с требуемым числом каналов
class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)

# Общее количество параметров модели
def count_model_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
