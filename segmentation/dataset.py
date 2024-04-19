LABELS = {
    0: "background",
    1: "liver",
}
COLORMAP = [
    [0, 0, 0],
    [255, 255, 255],
]

from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        # Загрузка изображения и маски
        img = Image.open(image_path).convert("RGB")
        target = Image.open(mask_path).convert("RGB")

        # Применение преобразований, если они заданы
        if self.transform:
            augmented = self.transform(image=np.array(img), mask=np.array(target))
            img, target = augmented["image"], augmented["mask"]

        # Преобразование в тензоры и возврат кортежа
        return img.float(), target.float().permute(2, 0, 1)