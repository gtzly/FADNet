import os
import csv
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from lib.utils.heatmap_decoder import get_transfrom_matrix
from lib.utils.paramlist import ParamsList

class KITTI(Dataset):
    def __init__(self, cfg, transforms):
        super(KITTI, self).__init__()
        self.root = cfg.data_root
        self.image_dir = os.path.join(self.root, "image_2")
        self.calib_dir = os.path.join(self.root, "calib")
        self.transforms = transforms
        self.image_files = sorted(list(os.listdir(self.image_dir)))
        self.image_files=self.image_files[:300]
        self.num_samples = len(self.image_files)
        self.classes = cfg.det_classes
        self.num_classes = len(self.classes)

        self.input_width = cfg.input_width
        self.input_height = cfg.input_height
        self.output_width = self.input_width // cfg.backbone_stride
        self.output_height = self.input_height // cfg.backbone_stride

        print("Totally {} files loaded.".format(self.num_samples))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        original_idx = self.image_files[idx].replace(".png", "")
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        img = Image.open(img_path)
        calib = self.get_calib(idx)

        center = np.array([i / 2 for i in img.size], dtype=np.float32)
        size = np.array([i for i in img.size], dtype=np.float32)
        center_size = [center, size]
        trans_affine = get_transfrom_matrix(
            center_size,
            [self.input_width, self.input_height]
        )
        trans_affine_inv = np.linalg.inv(trans_affine)
        img = img.transform(
            (self.input_width, self.input_height),
            method=Image.AFFINE,
            data=trans_affine_inv.flatten()[:6],
            resample=Image.BILINEAR,
        )
        trans_mat = get_transfrom_matrix(
            center_size,
            [self.output_width, self.output_height]
        )

        meta_data = ParamsList(image_size=size)
        meta_data.add_field("trans_mat", trans_mat)
        meta_data.add_field("calib", calib)

        img, meta_data = self.transforms(img, meta_data)

        return img, meta_data, original_idx


    def get_calib(self, idx):
        file_name = self.image_files[idx].replace(".png", ".txt")
        with open(os.path.join(self.calib_dir, file_name), 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=' ')
            for line, row in enumerate(reader):
                if row[0] == 'P2:':
                    calib = row[1:]
                    calib = [float(i) for i in calib]
                    calib = np.array(calib, dtype=np.float32).reshape(3, 4)
                    calib = calib[:3, :3]
                    break
        return calib
