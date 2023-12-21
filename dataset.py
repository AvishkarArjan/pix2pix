from PIL import Image
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
import config


class MapDataset(Dataset):
    def __init__(self, input_dir, target_dir):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.in_list_files = os.listdir(self.input_dir)
        self.target_list_files = os.listdir(self.target_dir)

    def __len__(self):
        return len(self.in_list_files)

    def __getitem__(self, index):
        in_img_file = self.in_list_files[index]

        # target_img_file: str
        # for tar_img in self.target_list_files:
        #     if tar_img == in_img_file:
        #         target_img_file = tar_img
        #         break

        target_img_file = self.target_list_files[index]
        # print(in_img_file, target_img_file)
        in_img_path = os.path.join(self.input_dir, in_img_file)
        target_img_path = os.path.join(self.target_dir, target_img_file)

        in_img = np.array(Image.open(in_img_path))
        tar_img = np.array(Image.open(target_img_path))

        augmentations = config.both_transform(image=in_img, image0=tar_img)

        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_mask(image=target_image)["image"]

        return input_image, target_image


if __name__ == "__main__":
    dataset = MapDataset("data/train/")
    loader = DataLoader(dataset, batch_size=5)
    for x, y in loader:
        print(x.shape)
        save_image(x, "x.png")
        save_image(y, "y.png")
        import sys

        sys.exit()
