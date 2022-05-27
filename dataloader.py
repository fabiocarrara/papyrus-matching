from dataclasses import dataclass

import numpy as np
from skimage import draw
from skimage.io import imread
from torch.utils.data import Dataset


@dataclass
class Match:
    y: int
    x: int
    rows: int
    cols: int
    patch_size: int
    gap: float

    def left_region(self):
        start = (self.y, self.x)
        end = (self.y + self.rows * self.patch_size, self.x + self.cols * self.patch_size)
        return start, end
    
    def right_region(self):
        offset = self.cols * self.patch_size + round(self.gap * self.patch_size)
        start = (self.y, self.x + offset)
        end = (self.y + self.rows * self.patch_size, self.x + self.cols * self.patch_size + offset)
        return start, end

    @staticmethod
    def availability(im, start, end):
        top, left = start
        bottom, right = end

        integral = im[bottom, right] - im[bottom, left] - im[top, right] + im[top, left]
        height, width = bottom - top + 1, right - left + 1

        return integral / (height * width)

    def is_good(self, im, threshold=0.7):
        left_start, left_end = self.left_region()
        right_start, right_end = self.right_region()

        height, width = im.shape
        if not (right_end[0] < height and right_end[1] < width):
            return False

        available_1 = self.availability(im, left_start, left_end)
        available_2 = self.availability(im, right_start, right_end)

        return (available_1 >= threshold) and (available_2 >= threshold)
    
    def draw(self, im, color):
        left_start, left_end = self.left_region()
        right_start, right_end = self.right_region()

        coords = draw.rectangle(left_start, left_end, shape=im.shape)
        draw.set_color(im, coords, color, 0.3)

        coords = draw.rectangle(right_start, right_end, shape=im.shape)
        draw.set_color(im, coords, color, 0.3)


class PapyrMatchesDataset(Dataset):
    def __init__(
        self,
        image_path,
        patch_size=64,
        stride=64,
        rows=4,
        cols=2,
        min_gap=0.125,  # minimum gap between left and right parts in fraction of patch size
        max_gap=1.500,  # maximum gap between left and right parts in fraction of patch size
        step_gap=0.125, # step for exploring gap values
        min_availability=0.7,  # minimum fraction of non-missing papyrus to consider a region usable
        transform=None
    ):
        self.image_path = image_path
        self.patch_size = patch_size
        self.stride = stride
        self.rows = rows
        self.cols = cols
        self.min_gap = min_gap
        self.max_gap = max_gap
        self.step_gap = step_gap
        self.min_availability = min_availability
        self.transform = transform
    
        # load image data
        self.image = imread(image_path)  # RGBA image
        self.image[:, :, -1] = np.where(self.image[:, :, -1] >= 127, np.uint8(255), np.uint8(0))  # binarize alpha

        # matches data
        self.matches = self._find_matches()
    
    def _find_matches(self):
        integral_mask = self.image[:, :, -1].astype(np.float32).cumsum(axis=0).cumsum(axis=1) / 255.

        matches = []
        height, width = self.image.shape[:2]
        for gap in np.arange(self.min_gap, self.max_gap + 0.001, self.step_gap):
            for y in range(0, height, self.stride):
                for x in range(0, width, self.stride):
                    match = Match(y, x, self.rows, self.cols, self.patch_size, gap)
                    if match.is_good(integral_mask, threshold=self.min_availability):
                        matches.append(match)
        
        return matches
    
    def __len__(self):
        return len(self.matches)
    
    def __getitem__(self, index):
        match = self.matches[index]

        (l_y0, l_x0), (l_y1, l_x1) = match.left_region()
        (r_y0, r_x0), (r_y1, r_x1) = match.right_region()

        left_region = self.image[l_y0:l_y1, l_x0:l_x1, :]
        right_region = self.image[r_y0:r_y1, r_x0:r_x1, :]

        if self.transform:
            left_region = self.transform(left_region)
            right_region = self.transform(right_region)

        return left_region, right_region