# Code taken from here:
# https://github.com/Project-MONAI/tutorials/blob/main/2d_segmentation/torch/unet_training_dict.py
# Modified the input data to suit my project

import logging
import os
import sys
import tempfile
from glob import glob

import monai.config
import monai.data
import torch
from PIL import Image
import pydicom
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import create_test_image_2d, list_data_collate, decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd,
)
from monai.visualize import plot_2d_or_3d_image

import prepdata
from pdb import set_trace

import matplotlib.pyplot as plt

def main(tempdir, patient_num: int):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    print(f"generating synthetic data to {tempdir} (this may take a while)")
    all_dicom_images = prepdata.load_dicom_images(r'C:/Users/mrtan/Desktop/Job Stuff/Sum2024Research/proj/newpatients/p'+str(patient_num)+'/')
    image_shape = all_dicom_images.shape[1:]

    pairs = prepdata.csv_extractor()
    all_masks = prepdata.create_masks_from_convex_hull(image_shape, pairs)

    for i, data in enumerate(all_dicom_images):
        Image.fromarray((all_dicom_images[i] * 255).astype("uint8")).save(os.path.join(tempdir, f"img{i:d}.png"))
        Image.fromarray((all_masks[i] * 255).astype("uint8")).save(os.path.join(tempdir, f"seg{i:d}.png"))
    images = sorted(glob(os.path.join(tempdir, "img*.png")))
    segs = sorted(glob(os.path.join(tempdir, "seg*.png")))
    test_files = [{"img": img, "seg": seg} for img, seg in zip(images[25:], segs[25:])]

    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            ScaleIntensityd(keys=["img", "seg"]),
        ]
    )
    # define dataset, data loader
    check_ds = monai.data.Dataset(data=test_files, transform=val_transforms)
    # use batch_size=2 to loplad images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    check_loader = DataLoader(check_ds, batch_size=2, num_workers=4, collate_fn=list_data_collate)
    check_data = monai.utils.misc.first(check_loader)
    print(check_data["img"].shape, check_data["seg"].shape)

    # create a test dataset
    test_ds = monai.data.Dataset(data=test_files, transform=val_transforms)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=True, num_workers=4, collate_fn=list_data_collate)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    model.load_state_dict(torch.load('runs/cluster_runs/Jul_10_24_00-53/best_metric_model_segmentation2d_dict.pth'))
    model.eval()
    with torch.no_grad():
        test_data = next(iter(test_loader))
        test_image, test_label = test_data["img"].to(device), test_data["seg"].to(device)
        roi_size = (96, 96)
        sw_batch_size = 4
        test_output = sliding_window_inference(test_image, roi_size, sw_batch_size, model)
        test_output = [post_trans(i) for i in decollate_batch(test_output)]
    set_trace()
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(test_image[0][0].cpu().detach().numpy())
    ax[1].imshow(test_output[0][0].cpu().detach().numpy())
    plt.show()

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tempdir:
        main(tempdir, 301)
    








