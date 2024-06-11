from matplotlib.colors import ListedColormap
import numpy as np
import os
from pathlib import Path 

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms import v2 as transformsv2
from model import UNet
from tqdm import tqdm
import tifffile
from data_processing import SDTDataset, GradientDataset
import sys
from functools import partial
import argparse
sys.path.append('.')

def salt_and_pepper_noise(image, amount=0.05):
    """
    Add salt and pepper noise to an image
    """
    out = image.clone()
    num_salt = int(amount * image.numel() * 0.5)
    num_pepper = int(amount * image.numel() * 0.5)

    # Add Salt noise
    coords = [
        torch.randint(0, i - 1, [num_salt]) if i > 1 else [0] * num_salt
        for i in image.shape
    ]
    out[coords] = 1

    # Add Pepper noise
    coords = [
        torch.randint(0, i - 1, [num_pepper]) if i > 1 else [0] * num_pepper
        for i in image.shape
    ]
    out[coords] = 0

    return out

def gaussian_noise(image, mean = 0, var = 0.0001):
    ch, row,col= image.shape
    mean = mean
    var = var
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(ch,row,col))
    gauss = gauss.reshape(ch,row,col)
    noisy = image + gauss
    noisy = transforms.Normalize([0.5], [0.5])(noisy)
    return noisy.to(torch.float)


def train(
    model,
    loader,
    optimizer,
    loss_function,
    epoch,
    log_interval=100,
    log_image_interval=20,
    tb_logger=None,
    device=None,
    ignore_background=False,
):
    if device is None:
        # You can pass in a device or we will default to using
        # the gpu. Feel free to try training on the cpu to see
        # what sort of performance difference there is
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    # set the model to train mode
    model.train()

    # move model to device
    model = model.to(device)

    # iterate over the batches of this epoch
    for batch_id, batch in enumerate(loader):
        if ignore_background:
            x, y, ignore_mask = batch[0], batch[1], batch[2]
            x, y, ignore_mask = x.to(device), y.to(device), ignore_mask.to(device)
        else:
            x, y = batch[0], batch[1]
            x, y = x.to(device), y.to(device)

        # zero the gradients for this iteration
        optimizer.zero_grad()

        # apply model and calculate loss
        prediction = model(x)
        if prediction.shape != y.shape:
            y = crop(y, prediction)
        if y.dtype != prediction.dtype:
            y = y.type(prediction.dtype)

        # mask out areas that should be ignored in gt and prediction by multiplying by ignore mask
        if ignore_background:
            y = y * ignore_mask
            prediction = prediction * ignore_mask

        loss = loss_function(prediction, y)

        # backpropagate the loss and adjust the parameters
        loss.backward()
        optimizer.step()

        # log to console
        if batch_id % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_id * len(x),
                    len(loader.dataset),
                    100.0 * batch_id / len(loader),
                    loss.item(),
                )
            )

        # log to tensorboard
        if tb_logger is not None:
            step = epoch * len(loader) + batch_id
            tb_logger.add_scalar(
                tag="train_loss", scalar_value=loss.item(), global_step=step
            )
            # check if we log images in this iteration
            if step % log_image_interval == 0:
                tb_logger.add_images(
                    tag="input", img_tensor=x.to("cpu"), global_step=step
                )
                tb_logger.add_images(
                    tag="target", img_tensor=y.to("cpu"), global_step=step
                )
                tb_logger.add_images(
                    tag="prediction",
                    img_tensor=prediction.to("cpu").detach(),
                    global_step=step,
                )


def validate(model,
    loader,
    loss_function,
    epoch,
    tb_logger=None,
    device='cuda'):
    
    model.eval()
    running_loss = 0.

    i = 0
    with torch.no_grad():
        for batch_id, batch in enumerate(loader):
            # move input and target to the active device (either cpu or gpu)
            x, y = batch[0], batch[1]
            x, y = x.to(device), y.to(device)
            pred = model(x)
            val_loss = loss_function(pred, y)
            running_loss += val_loss
            i += 1
    val_loss = running_loss / i
    if tb_logger is not None:
        tb_logger.add_scalar("Loss/validation", val_loss, epoch)
    print(f"Validation loss after training epoch {epoch} is {val_loss}")
    
    return val_loss



class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def main():
    device = "cuda"  # 'cuda', 'cpu', 'mps'
    assert torch.cuda.is_available()

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),  
            transforms.RandomRotation([90,90]),
            transforms.RandomCrop(256)   
            ]
    )

    img_transforms = transforms.Compose(
        [
            #transforms.GaussianBlur(kernel_size=5, sigma=3),
            #transformsv2.Lambda(salt_and_pepper_noise),
            transformsv2.Lambda(gaussian_noise)
        ]
    )
    img_transforms = None
    center_crop = True  # whether to do a center crop
    pad = 256  # min size in either dimension, will pad smaller images up to this size
    watershed_scale = 5  # scale over which to calculate the distance transform

    print("Loading data ...")

    train_data = SDTDataset(root_dir=user_rootdir, transform=transform, img_transform=img_transforms, train=True, ignore_background=ignore_background, 
                            center_crop=center_crop, pad=pad, watershed_scale=watershed_scale)
    train_loader = DataLoader(train_data, batch_size=10, shuffle=True, num_workers=8)
    val_data = SDTDataset(root_dir=user_rootdir, transform=None, img_transform=None, train=False, return_mask=False, ignore_background=False, 
                          center_crop=center_crop, pad=pad, mean=train_data.mean, std=train_data.std, watershed_scale=watershed_scale)
    val_loader = DataLoader(val_data, batch_size=10)

    """
    train_data = GradientDataset(transform=transform, img_transform=img_transforms, train=True, ignore_background=ignore_background, 
                            center_crop=center_crop, pad=pad)
    train_loader = DataLoader(train_data, batch_size=5, shuffle=True, num_workers=8)
    val_data = GradientDataset(transform=None, img_transform=None, train=False, ignore_background=False, 
                          center_crop=center_crop, pad=pad, mean=train_data.mean, std=train_data.std)
    val_loader = DataLoader(val_data, batch_size=5)
    """

    print(len(train_loader), len(val_loader))
    # Initialize the model.
    unet = UNet(
        depth=user_depth,
        in_channels=user_inCh,
        out_channels=user_outCh,
        final_activation="Tanh",
        num_fmaps=user_num_fmaps,
        fmap_inc_factor=user_fmap_inc_factor,
        downsample_factor=2,
        padding=user_padding,
        upsample_mode="nearest",
    )

    learning_rate = 1e-4
    loss = torch.nn.BCELoss()
    writer = SummaryWriter()
    optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, min_lr=1e-8)

    early_stopper = EarlyStopper(patience=20)
    for epoch in tqdm(range(user_epochs)):
        train(
            unet,
            train_loader,
            optimizer,
            loss,
            epoch,
            log_interval=10,
            device=device,
            tb_logger=writer
        )
        #TODO I think this is wrong, it is calculating the loss using the validation dataset?
        val_loss = validate(unet, 
                            val_loader,
                            loss,
                            epoch, 
                            writer,
                            device='cuda')


        scheduler.step(val_loss)
        writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], epoch)
        if early_stopper.early_stop(val_loss):
            print("Stopping test early!")
            break

        if epoch % 10 == 0:
            output_path = Path("logs/")
            output_path.mkdir(exist_ok=True)
            torch.save(
            {'model_state_dict': unet.state_dict()},
            "logs/tmp.pth"
            )

    output_path = Path("logs/")
    output_path.mkdir(exist_ok=True)
    torch.save(
        {'model_state_dict': unet.state_dict()},
        "logs/model.pth"
    )

    writer.flush()


if __name__ == "__main__":

    config = {}

    # Open the file in read mode
    with open("./config.txt", 'r') as file:
    # Read each line in the file
        for line in file:
            # Strip any leading/trailing whitespace and split the line into variable name and value
            name, value = line.strip().split('=')
            # Store the variable name and value in the dictionary
            config[name.strip()] = (value.strip())  

    # Access the values by variable names
    user_rootdir = config['rootdir'] # Full path to data folder. This data folder must contain two folders: one named 'train', containing the data to train on, and one named 'validate', to validate the model. Each of these folders must contain two additional folders in them: one folder named 'img', containing the raw data to train on, and one folder named 'mask', containing the segmented images. Ensure the raw images and the segmented outputs have the exact same name.
    user_depth = int(config['depth']) #UNet parameter: Depth of the Unet (integer)
    user_inCh = int(config['inCh']) #UNet parameter: Number of channels in the input raw images
    user_outCh = int(config['outCh']) #Unet parameter: Number of channels in the expected output
    user_num_fmaps = int(config['num_fmaps'])  #UNet parameter: Number of feature maps
    user_fmap_inc_factor = int(config['fmap_inc_factor']) #UNet parameter: Increment to the number of feature maps per convolutional layer
    user_padding = config['padding'] #UNet parameter. Padding used during convolution. Should be "same" or "valid"
    user_epochs = int(config['epochs']) #UNet hyperparameter: number of epochs. TODO: missing input for this


    main()
