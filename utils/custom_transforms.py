import torch
import numpy as np

torch.manual_seed(42)

class Mask:
    def __init__(self, mask_ratio=0.2, mask_method="grid"):
        self.mask_ratio = mask_ratio

        mask_factory = {
            "grid": self._apply_grid_mask,
            "pixel": self._apply_pixel_mask,
            "random_erasing": self._apply_random_erasing
        }
        self.mask_method = mask_factory.get(mask_method, self._apply_grid_mask)

    def __call__(self, image_tensor):
        masked = self.mask_method(image_tensor, self.mask_ratio)
        return masked, image_tensor

    def _apply_grid_mask(self, image_tensor, ratio):
        c, h, w = image_tensor.size()
        mask_height, mask_width = h // 14, w // 14
        grid_mask = torch.ones_like(image_tensor, device=image_tensor.device)
        rands = torch.rand((14, 14), device=image_tensor.device)

        for i in range(14):
            for j in range(14):
                if rands[i, j] < ratio:
                    y1, x1 = i * mask_height, j * mask_width
                    y2, x2 = min((i + 1) * mask_height,
                                 h), min((j + 1) * mask_width, w)
                    grid_mask[:, y1:y2, x1:x2] = 0

        return image_tensor * grid_mask

    def _apply_pixel_mask(self, image_tensor, ratio):
        mask = torch.rand_like(image_tensor) > ratio
        return image_tensor * mask

    def _apply_random_erasing(self, image_tensor, ratio):
        c, h, w = image_tensor.size()
        erase_area_max = ratio * h * w
        aspect_ratio = np.random.uniform(0.3, 3.0)

        erase_area = np.random.uniform(0.1 * erase_area_max, erase_area_max)
        erase_height = int(np.sqrt(erase_area / aspect_ratio))
        erase_width = int(erase_height * aspect_ratio)

        erase_height = min(erase_height, h)
        erase_width = min(erase_width, w)

        x = np.random.randint(0, w - erase_width)
        y = np.random.randint(0, h - erase_height)

        output = image_tensor.clone()
        output[:, y:y + erase_height, x:x +
                     erase_width] = torch.zeros((c, erase_height, erase_width))

        return output


class ScaleTrimap:
    def __call__(self, image_tensor):
        return (torch.round(image_tensor * 255., decimals=0) - 1).to(torch.long)
