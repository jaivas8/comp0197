from torchvision.datasets import VisionDataset
from PIL import Image
import os


class CocoImages(VisionDataset):
    def __init__(self, image_root, transform=None, split="train"):
        super().__init__(image_root, transform=transform)
        self.img_keys = sorted([int(n.split(".")[0])
                               for n in os.listdir(self.root)])
        num_imgs = len(self.img_keys)
        split_idx = int(0.5*num_imgs)
        if split == "train":
            self.img_keys = self.img_keys[:split_idx]
        elif split == "val":
            self.img_keys = self.img_keys[split_idx:]
        elif split != "all":
            raise Exception(
                "[CocoImages] Split argument must be str in ['train', 'val', 'all']")

    def _load_image(self, img_id: int):
        filename = f"{img_id:0>12}.jpg"
        return Image.open(os.path.join(self.root, filename)).convert("RGB")

    def __getitem__(self, index: int):
        image = self._load_image(self.img_keys[index])
        if self.transforms is not None:
            image = self.transform(image)
        return image

    def __len__(self) -> int:
        return len(self.img_keys)
