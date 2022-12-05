import os
import sys
import torch
import logging
import numpy as np
from PIL import Image
from PIL import ImageOps
        
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import RandomCrop, ToTensor, Compose, CenterCrop, RandomHorizontalFlip, RandomVerticalFlip


class ImageDataLoader():
    def __init__(self, config):
        # h = config.U + 1
        # w = config.L + config.R + 1
        train_datas = [config.train_data_1]
        train_datas =  train_datas[0:config.num_train_dirs]
        self.train_dataset = ImageDataset(train_datas,
                                          config.patch_size,
                                          train=True)
        self.test_dataset  = ImageDataset_test(config.test_data,
                                            config.test_patch_size,
                                          train=False)
        # self.valid_dataset = ImageDataset(config.valid_data,
        #                                     config.val_patch_size,
        #                                     train=False)

        self.valid_dataset = ImageDataset_test(config.test_data,
                                            config.test_patch_size,
                                          train=False)
        
        num_workers = 0 if config.mode == 'debug' else 4 # 0 # 4

        self.train_loader = DataLoader(self.train_dataset, 
                                        batch_size=config.batch_size, 
                                        shuffle=True, 
                                        num_workers=num_workers, 
                                        pin_memory=True, 
                                        drop_last=False)

        self.test_loader  = DataLoader(self.test_dataset, 
                                        batch_size=1, #5,
                                        shuffle=False, 
                                        num_workers=0, #num_workers,
                                        pin_memory=False, #True,
                                        drop_last=False)

        # self.valid_loader = DataLoader(self.valid_dataset,
        #                                 batch_size=config.val_batch_size,
        #                                 shuffle=False,
        #                                 num_workers=0,
        #                                 pin_memory=False,
        #                                 drop_last=False)
        self.valid_loader = DataLoader(self.test_dataset,
                                        batch_size=1, #5,
                                        shuffle=False,
                                        num_workers=0, #num_workers,
                                        pin_memory=False, #True,
                                        drop_last=False)


class ImageDataset(Dataset):
    def __init__(self, root, size, train=True):
        self.size = size # fatih added this to know size in getitem to resize there wif image is smaller than size
        try:
            if isinstance(root, str):
                self.image_files = [os.path.join(root, f)  for f in os.listdir(root) if (f.endswith('.png') or f.endswith('.jpg'))]
            else:
                self.image_files = []
                for i in range(0, len(root)):
                    self.image_files_temp = [os.path.join(root[i], f)  for f in os.listdir(root[i]) if (f.endswith('.png') or f.endswith('.jpg'))]
                    self.image_files = self.image_files + self.image_files_temp
        except:
            logging.getLogger().exception('Dataset could not be found. Drive might be unmounted.', exc_info=False)
            sys.exit(1)
        if size == 0:
            self.transforms = Compose([ToTensor()])
        else:
            crop = RandomCrop(size) if train else CenterCrop(size)
            self.transforms = Compose([crop,  ToTensor()])
            # self.transforms = Compose([crop, RandomHorizontalFlip(), RandomVerticalFlip(), ToTensor()])

    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, i):
        # NOTE check if the output is normalized between 0-1
        # img = pil_loader(self.image_files[i])
        # return self.transforms(img)
        img = pil_loader(self.image_files[i])
        # check size of image and resize it if width or height less than requested size
        width, height = img.size
        ws, hs = 0, 0
        if width < self.size and self.size > 0:
            ws = 1
        if height < self.size and self.size > 0:
            hs = 1
        if ws == 1 and hs == 1:
            img = ImageOps.fit(img, self.size)
        elif ws == 1 and hs == 0:
            img = ImageOps.fit(img, (self.size, height))
        elif ws == 0 and hs == 1:
            img = ImageOps.fit(img, (width, self.size))
        return self.transforms(img)

class ImageDataset_test(Dataset):
    def __init__(self, root, size, train=False):
        self.size = size  # fatih added this to know size in getitem to resize there wif image is smaller than size
        try:
            if isinstance(root, str):
                self.image_files = [os.path.join(root, f) for f in os.listdir(root) if
                                    (f.endswith('.png') or f.endswith('.jpg'))]
            else:
                self.image_files = []
                for i in range(0, len(root)):
                    self.image_files_temp = [os.path.join(root[i], f) for f in os.listdir(root[i]) if
                                             (f.endswith('.png') or f.endswith('.jpg'))]
                    self.image_files = self.image_files + self.image_files_temp
        except:
            logging.getLogger().exception('Dataset could not be found. Drive might be unmounted.', exc_info=False)
            sys.exit(1)
        if size == 0:
            self.transforms = Compose([ToTensor()])
        else:
            crop = CenterCrop(size)
            # self.transforms = Compose([crop,  ToTensor()])
            self.transforms = Compose([crop, ToTensor()])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, i):
        # NOTE check if the output is normalized between 0-1
        # img = pil_loader(self.image_files[i])
        # return self.transforms(img)
        img = pil_loader(self.image_files[i])
        # check size of image and resize it if width or height less than requested size
        width, height = img.size
        ws, hs = 0, 0
        if width < self.size and self.size > 0:
            ws = 1
        if height < self.size and self.size > 0:
            hs = 1
        if ws == 1 and hs == 1:
            img = ImageOps.fit(img, self.size)
        elif ws == 1 and hs == 0:
            img = ImageOps.fit(img, (self.size, height))
        elif ws == 0 and hs == 1:
            img = ImageOps.fit(img, (width, self.size))
        return self.transforms(img)

def pil_loader(path):
    # open path as file to avoid ResourceWarning 
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
