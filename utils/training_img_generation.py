import os
import sys
from PIL import Image
from PIL import ImageOps
from torchvision.transforms import RandomCrop, ToTensor, Compose, CenterCrop, RandomHorizontalFlip, RandomVerticalFlip


# create new training set with small res images that are cropped and saved from lareg images
def save_patches_from_imgs(patch_size=256, save_percentage=25, min_num_patch_per_img=9):
    # dirs to read from and to write to
    root_rd = "/media/research/DL-Pytorch-1/CLIC/train_mobile_2020"
    root_wr = "/media/research/DL-Pytorch-1/CLIC/train_mobile_2020_256x256"
    # operations to perform
    # transforms = Compose([RandomCrop(patch_size), RandomHorizontalFlip(), RandomVerticalFlip()])  # , ToTensor()
    transforms = Compose([RandomCrop(patch_size)])  # dont flip here, flip online in dataloader
    # get all file names in rd directory
    try:
        image_files = [os.path.join(root_rd, f) for f in os.listdir(root_rd) if (f.endswith('.png') or f.endswith('.jpg'))]
        image_names = [f for f in os.listdir(root_rd) if (f.endswith('.png') or f.endswith('.jpg'))]
        print('Read directory ' + root_rd + ' has ' + str(len(image_files)) + ' files.')
    except:
        print('Read directory could not be found. Exiting.')
        sys.exit(1)
    # check num files in write directory and create it if not exist
    if os.path.exists(root_wr):
        image_files_wr = [os.path.join(root_wr, f) for f in os.listdir(root_wr) if (f.endswith('.png') or f.endswith('.jpg'))]
        print('Write directory ' + root_wr + ' has ' + str(len(image_files_wr)) + ' jpg or png files. Files may be overwritten if names collide.')
        inp = input("Type yes to proceed...")
        if inp != 'yes':
            print("OK then, exiting.")
            sys.exit(1)
    else:
        print('Write directory could not be found. Trying to create one...')
        try:
            os.mkdir(root_wr)
            print('Write directory created: ' + root_wr)
        except:
            print('Write directory could not be created. Exiting.')
            sys.exit(1)
    print('Starting process...')
    # go over all images
    for img_idx, img_file in enumerate(image_files):
        img = pil_loader(img_file)
        # check size of image and resize it if width or height less than requested size
        width, height = img.size
        ws, hs = 0, 0
        if width < patch_size and patch_size > 0:
            ws = 1
        if height < patch_size and patch_size > 0:
            hs = 1
        if ws == 1 and hs == 1:
            img = ImageOps.fit(img, patch_size)
        elif ws == 1 and hs == 0:
            img = ImageOps.fit(img, (patch_size, height))
        elif ws == 0 and hs == 1:
            img = ImageOps.fit(img, (width, patch_size))
        # determine number of patches to save from this image
        num_patches_per_img = int(1 + save_percentage/100 * (width*height / patch_size**2))
        if num_patches_per_img < min_num_patch_per_img:
            num_patches_per_img = min_num_patch_per_img
        # get that many patches and save them
        for pn in range(0, num_patches_per_img):
            patch = transforms(img)
            imgn = image_names[img_idx].split('.')
            patch_name = imgn[0] + str("_rndcrop_{:d}x{:d}_{:02d}.".format(patch_size, patch_size, pn)) + imgn[1]
            patch.save(os.path.join(root_wr, patch_name), "PNG")


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        rgb_img = img.convert('RGB')
        return rgb_img
