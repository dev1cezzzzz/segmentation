from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import os
from torch import Tensor

class CustomDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.image_paths = []
        self.mask_paths = []

        # Получаем список файлов в папке Links
        link_dir = os.path.join(root_dir, 'images')
        mask_dir = os.path.join(root_dir, 'masks')

        # Перебираем файлы и добавляем изображения и соответствующие маски в списки
        for file in os.listdir(link_dir):
            if file.endswith(".png"):  # Предполагаем, что изображения имеют расширения .png
                image_path = os.path.join(link_dir, file)
                mask_file = "mask_" + file
                mask_path = os.path.join(mask_dir, mask_file)
                if os.path.exists(mask_path):
                    self.image_paths.append(image_path)
                    self.mask_paths.append(mask_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        img = Image.open(self.image_paths[index]).convert("RGB")
        target = Image.open(self.mask_paths[index]).convert("L")

        img = np.array(img, dtype=np.uint8)
        target = np.array(target, dtype=np.uint8)

        assert self.transforms is not None

        augmented = self.transforms(image=img, mask=target)
        return augmented["image"].float(), augmented["mask"].unsqueeze(0).float()
