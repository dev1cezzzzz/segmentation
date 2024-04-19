

LABELS_MAP = {
    0: "background",
    1: "liver",
}


COLORMAP = [
    [0, 0, 0],
    [128, 0, 0],
]

import os
from PIL import Image
import numpy as np
from torch import Tensor

class CustomVOCSegmentation:
    def __init__(self, root: str, image_set: str = "train", transform=None, target_transform=None,
                 transforms=None):
        super().__init__(root, image_set, transform, target_transform, transforms)

        self.images_folder = os.path.join(root, "images")  # Папка с изображениями
        self.masks_folder = os.path.join(root, "masks")  # Папка с масками

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        img_path = os.path.join(self.images_folder, self.images[index])
        target_path = os.path.join(self.masks_folder, self.masks[index])

        img = Image.open(img_path).convert("RGB")
        target = Image.open(target_path).convert("RGB")

        img = np.array(img, dtype=np.uint8)
        target = np.array(target, dtype=np.uint8)
        target = self._convert_to_segmentation_mask(target)

        assert self.transform is not None

        augmented = self.transform(image=img, mask=target)
        return augmented["image"].float(), augmented["mask"].float().permute(2, 0, 1)

    @staticmethod
    def _convert_to_segmentation_mask(mask: np.ndarray) -> np.ndarray:
        height, width = mask.shape[:2]
        segmentation_mask = np.zeros((height, width, len(VOC_COLORMAP)), dtype=np.uint8)
        for label_index, label in enumerate(VOC_COLORMAP):
            segmentation_mask[:, :, label_index] = np.all(mask == label, axis=-1).astype(np.uint8)

        return segmentation_mask
