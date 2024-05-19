import numpy as np
from scipy.ndimage import distance_transform_edt
from matplotlib.colors import ListedColormap
import os
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import tifffile
from scipy.ndimage import binary_fill_holes
from torchvision.transforms import v2 as transformsv2
from pathlib import Path
from skimage.filters.rank import gradient
from skimage.morphology import disk


def compute_sdt(labels: np.ndarray, scale: int = 5):
    """Function to compute a signed distance transform."""

    # compute the distance transform inside and outside of the objects
    labels = np.asarray(labels)
    ids = np.unique(labels)
    ids = ids[ids != 0]
    inner = np.zeros(labels.shape, dtype=np.float32)

    for id_ in ids:
        inner += distance_transform_edt(labels == id_)
    outer = distance_transform_edt(labels == 0)

    # create the signed distance transform
    distance = inner - outer

    # scale the distances so that they are between -1 and 1 (hint: np.tanh)
    distance = np.tanh(distance / scale)

    # be sure to return your solution as type 'float'
    return distance.astype(float)


class SDTDataset(Dataset):
    """A PyTorch dataset to load images and cell masks."""

    def __init__(self, root_dir = "/group/dl4miacourse/projects/membrane/ecad_gfp_cropped/", 
    transform=None, img_transform=None, return_mask=False, train=False, ignore_background=False, center_crop=True, 
    pad=0, mean=None, std=None, watershed_scale=5):
        
        # the directory with all the training samples
        if train:
            self.root_dir = root_dir + 'train/'
        else:
            self.root_dir = root_dir + 'test/'
        
        #the name of the raw images is different from the name of the mask names so we temporarilly store this info in separate variables
        self.list_images = os.listdir(self.root_dir+"/im/") 

        self.list_masks = os.listdir(self.root_dir+"/mask/") 

        assert len(self.list_images) == len(self.list_masks)

        #store information on the number of samples
        self.nsamples = len(self.list_images)
        self.seed = 0

        self.return_mask = return_mask
        self.ignore_background = ignore_background
        self.center_crop = center_crop
        self.pad = pad
        self.watershed_scale = watershed_scale

        self.transform = (
            transform  # transformations to apply to both inputs and targets
        )

        self.img_transform = img_transform  # transformations to apply to raw image only
        #  transformations to apply just to inputs
        inp_transforms = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.ToTensor(),
            ]
        )

        self.loaded_imgs = [None] * self.nsamples
        self.loaded_masks = [None] * self.nsamples
        if self.ignore_background:
            self.ignore_masks = [None] * self.nsamples
        for sample_ind in range(self.nsamples):
            img_path = os.path.join(self.root_dir, "im", self.list_images[sample_ind])
            image = Image.open(img_path)
            image.load()

            embryo_info = self.list_images[sample_ind].split("_max_")[0]
            time_info = "_" + self.list_images[sample_ind].split("_")[-1]
            mask_filename = [i for i in self.list_masks if (embryo_info in i and time_info in i)][0]
            mask_path = os.path.join(self.root_dir, "mask", mask_filename)
            mask = Image.open(mask_path)
            mask.load()

            if self.ignore_background:
                ignore_mask = np.array(mask) > 0
                ignore_mask = binary_fill_holes(ignore_mask)  # so that we keep the wound area too
                ignore_mask = Image.fromarray(ignore_mask)

            if self.pad > 0:
                pad_dims = [self.pad - this_dim for this_dim in np.array(image).shape]
                for p, this_pad in enumerate(pad_dims):
                    if this_pad > 0:
                        pad_dims[p] = (this_pad // 2) + 1
                    else:
                        pad_dims[p] = 0
                pad_dims = [pad_dims[1], pad_dims[0]]  # convert from x,y to row,col
                image = transformsv2.Pad(pad_dims)(image)
                mask = transformsv2.Pad(pad_dims)(mask)
                if ignore_background:
                    ignore_mask = transformsv2.Pad(pad_dims)(ignore_mask)

            if self.center_crop:
                image = transforms.CenterCrop(256)(image)
                mask = transforms.CenterCrop(256)(mask)
                if ignore_background:
                    ignore_mask = transforms.CenterCrop(256)(ignore_mask)

            self.loaded_imgs[sample_ind] = inp_transforms(image)
            self.loaded_masks[sample_ind] = mask
            
            if self.ignore_background:
                self.ignore_masks[sample_ind] = ignore_mask

        #if you want to retrieve mean and sd from the train dataset
        if mean is None or std is None:
            img_array = np.concatenate([this_img.numpy().flatten() for this_img in self.loaded_imgs])
            self.mean = img_array.mean()
            self.std = img_array.std()
        else: 
            self.mean = mean
            self.std = std

        for i, img in enumerate(self.loaded_imgs):
            self.loaded_imgs[i] = transforms.Normalize([self.mean], [self.std])(img)


    # get the total number of samples
    def __len__(self):
        return self.nsamples

    # fetch the training sample given its index
    def __getitem__(self, idx):
        # We'll be using the Pillow library for reading files
        # since many torchvision transforms operate on PIL images
        image = self.loaded_imgs[idx]
        mask = self.loaded_masks[idx]
        if self.ignore_background:
            ignore_mask = self.ignore_masks[idx]
        if self.transform is not None:
            # Note: using seeds to ensure the same random transform is applied to
            # the image and mask
            self.seed = torch.seed()
            torch.manual_seed(self.seed)
            image = self.transform(image)
            torch.manual_seed(self.seed)
            mask = self.transform(mask)
            if self.ignore_background:
                torch.manual_seed(self.seed)
                ignore_mask = self.transform(ignore_mask)
        sdt = self.create_sdt_target(mask)
        if self.img_transform is not None:
            image = self.img_transform(image)

        # return stuff in order image, mask, sdt, ignore_mask but only if applicable
        return_images=[image]
        if self.return_mask:
            return_images.append(transforms.ToTensor()(mask))
        return_images.append(sdt)
        if self.ignore_background:
            return_images.append(transforms.ToTensor()(ignore_mask))
        
        return return_images

    def create_sdt_target(self, mask):

        sdt_target_array = compute_sdt(mask, scale=self.watershed_scale)
        sdt_target = transforms.ToTensor()(sdt_target_array)
        return sdt_target.float()

    def getImageList(self):
        return self.list_images

    def getMaskList(self):
        return self.list_masks


class GradientDataset(Dataset):
    """A PyTorch dataset to load images and cell masks."""

    def __init__(self, root_dir = "/group/dl4miacourse/projects/membrane/ecad_gfp_cropped/", 
    transform=None, img_transform=None, train=False, ignore_background=False, center_crop=True, 
    pad=0, mean=None, std=None):
        
        # the directory with all the training samples
        if train:
            self.root_dir = root_dir + 'train/'
        else:
            self.root_dir = root_dir + 'test/'
        
        #the name of the raw images is different from the name of the mask names so we temporarilly store this info in separate variables
        self.list_images = os.listdir(self.root_dir+"/im/") 

        self.list_masks = os.listdir(self.root_dir+"/mask/") 

        assert len(self.list_images) == len(self.list_masks)

        #store information on the number of samples
        self.nsamples = len(self.list_images)
        self.seed = 0

        self.ignore_background = ignore_background
        self.center_crop = center_crop
        self.pad = pad

        self.transform = (
            transform  # transformations to apply to both inputs and targets
        )

        self.img_transform = img_transform  # transformations to apply to raw image only
        #  transformations to apply just to inputs
        inp_transforms = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.ToTensor(),
            ]
        )

        self.loaded_imgs = [None] * self.nsamples
        self.loaded_masks = [None] * self.nsamples
        if self.ignore_background:
            self.ignore_masks = [None] * self.nsamples
        for sample_ind in range(self.nsamples):
            img_path = os.path.join(self.root_dir, "im", self.list_images[sample_ind])
            image = Image.open(img_path)
            image.load()

            embryo_info = self.list_images[sample_ind].split("_max_")[0]
            time_info = "_" + self.list_images[sample_ind].split("_")[-1]
            mask_filename = [i for i in self.list_masks if (embryo_info in i and time_info in i)][0]
            mask_path = os.path.join(self.root_dir, "mask", mask_filename)
            mask = Image.open(mask_path)
            mask.load()

            if self.ignore_background:
                ignore_mask = np.array(mask) > 0
                ignore_mask = binary_fill_holes(ignore_mask)  # so that we keep the wound area too
                ignore_mask = Image.fromarray(ignore_mask)
            
            # get the mask borders which will become our binary prediction mask
            grad_mask = gradient(np.array(mask), disk(3))
            mask = Image.fromarray(grad_mask > 0)

            if self.pad > 0:
                pad_dims = [self.pad - this_dim for this_dim in np.array(image).shape]
                for p, this_pad in enumerate(pad_dims):
                    if this_pad > 0:
                        pad_dims[p] = (this_pad // 2) + 1
                    else:
                        pad_dims[p] = 0
                pad_dims = [pad_dims[1], pad_dims[0]]  # convert from x,y to row,col
                image = transformsv2.Pad(pad_dims)(image)
                mask = transformsv2.Pad(pad_dims)(mask)
                if ignore_background:
                    ignore_mask = transformsv2.Pad(pad_dims)(ignore_mask)

            if self.center_crop:
                image = transforms.CenterCrop(256)(image)
                mask = transforms.CenterCrop(256)(mask)
                if ignore_background:
                    ignore_mask = transforms.CenterCrop(256)(ignore_mask)

            mask = transforms.ToTensor()(mask)
            self.loaded_imgs[sample_ind] = inp_transforms(image)
            self.loaded_masks[sample_ind] = mask
            
            if self.ignore_background:
                self.ignore_masks[sample_ind] = ignore_mask

        #if you want to retrieve mean and sd from the train dataset
        if mean is None or std is None:
            img_array = np.concatenate([this_img.numpy().flatten() for this_img in self.loaded_imgs])
            self.mean = img_array.mean()
            self.std = img_array.std()
        else: 
            self.mean = mean
            self.std = std

        for i, img in enumerate(self.loaded_imgs):
            self.loaded_imgs[i] = transforms.Normalize([self.mean], [self.std])(img)


    # get the total number of samples
    def __len__(self):
        return self.nsamples

    # fetch the training sample given its index
    def __getitem__(self, idx):
        # We'll be using the Pillow library for reading files
        # since many torchvision transforms operate on PIL images
        image = self.loaded_imgs[idx]
        mask = self.loaded_masks[idx]
        if self.ignore_background:
            ignore_mask = self.ignore_masks[idx]
        if self.transform is not None:
            # Note: using seeds to ensure the same random transform is applied to
            # the image and mask
            self.seed = torch.seed()
            torch.manual_seed(self.seed)
            image = self.transform(image)
            torch.manual_seed(self.seed)
            mask = self.transform(mask)
            if self.ignore_background:
                torch.manual_seed(self.seed)
                ignore_mask = self.transform(ignore_mask)
        if self.img_transform is not None:
            image = self.img_transform(image)

        # return stuff in order image, mask, sdt, ignore_mask but only if applicable
        return_images=[image, mask]
        if self.ignore_background:
            return_images.append(transforms.ToTensor()(ignore_mask))
        
        return return_images

    def getImageList(self):
        return self.list_images

    def getMaskList(self):
        return self.list_masks