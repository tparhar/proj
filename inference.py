import logging

import numpy as np
import torch
import pydicom
import matplotlib.pyplot as plt

import monai
from monai.data import pad_list_data_collate, decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    LoadImaged,
    Resized,
    ScaleIntensityRangePercentilesd,
)

import build_database
from pdb import set_trace

database_folder = 'imgs_and_segs'

def main():
    logging.getLogger('pydicom').setLevel(logging.WARNING)

    dataset = build_database.parse_database(database_folder)
    train_files, val_files, test_files = build_database.split_data(dataset)
    test_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            Resized(keys=["img", "seg"], spatial_size=[256, 256], mode=["bilinear", "nearest"]),
            ScaleIntensityRangePercentilesd(keys=["img"], lower=0, upper=100, b_min=0, b_max=1),
        ]
    )

    # create a validation data loader
    test_ds = monai.data.Dataset(data=test_files, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=6, collate_fn=pad_list_data_collate)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    model.load_state_dict(torch.load('runs/new_transforms/imgs_and_segs_lr_0.0001_epochs_150_batchsize_32/best_metric_model_segmentation2d_dict.pth'))
    model.eval()
    with torch.no_grad():
        test_images = None
        test_labels = None
        test_outputs = None
        for test_data in test_loader:
            test_images, test_labels = test_data["img"].to(device), test_data["seg"].to(device)
            roi_size = (96, 96)
            sw_batch_size = 4
            test_outputs = model(test_images)
            test_outputs = [post_trans(i) for i in decollate_batch(test_outputs)]
            # compute metric for current iteration
            dice_metric(y_pred=test_outputs, y=test_labels)
        # aggregate the final mean dice result
        metric = dice_metric.aggregate().item()
        dice_metric.reset()
        print("Dice Metric: {}".format(metric))
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(test_images[0][0].cpu().detach().numpy())
        ax[1].imshow(test_outputs[0][0].cpu().detach().numpy())
        plt.show()
if __name__ == "__main__":
    main()