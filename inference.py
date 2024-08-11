# Code taken from here:
# https://github.com/Project-MONAI/tutorials/blob/main/2d_segmentation/torch/unet_training_dict.py
# Modified the input data to suit my project

import os
import logging

import pandas as pd

import torch

import monai
from monai.data import list_data_collate, pad_list_data_collate, decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import ConfusionMatrixMetric, DiceMetric, MeanIoU
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
)
from monai.visualize import plot_2d_or_3d_image

import build_database
from pdb import set_trace

def main():
    logger = logging.getLogger('pydicom')
    logger.disabled = True

    database_folder = 'imgs_and_segs'

    dataset = build_database.parse_database(database_folder)
    train_files, val_files, test_files = build_database.split_data(dataset, onlyzeros=True, train_percent=0., val_percent=0., test_percent=1.)

    test_transforms = build_database.TransformDict('onlyzeros', [96, 96], 4).return_transform_dict()["test"]
    post_trans = Compose(
        [
            Activations(sigmoid=True),
            AsDiscrete(threshold=0.5)
        ]
    )

    test_ds = monai.data.Dataset(data=test_files, transform=test_transforms)
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        num_workers=0,
        collate_fn=list_data_collate
    )

    dice_metric = DiceMetric(include_background=False)

    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )

    patients_and_metrics = list()

    #Inference Section - this is the stuff that really matters
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    with torch.no_grad():
        test_images = None
        test_labels = None
        test_outputs = None
        for test_data in test_loader:
            test_images, test_labels = test_data["img"], test_data["seg"]
            test_outputs = sliding_window_inference(
                test_images,
                [96, 96],
                sw_batch_size=4,
                predictor=model,
                overlap=0.25
            )
            test_outputs = [post_trans(i) for i in decollate_batch(test_outputs)]
            # compute metric for current iteration
            dice_metric(y_pred=test_outputs, y=test_labels)
            metric = dice_metric.aggregate().item()
            dice_metric.reset()

            patient_file_path = test_images.meta.get("filename_or_obj")
            patient_num = os.path.basename(patient_file_path[0]).split('_')[0]
            patient_and_dice = [patient_num, metric]
            patients_and_metrics.append(patient_and_dice)
    
    columns = ['patient_number', 'dice']

    df = pd.DataFrame(patients_and_metrics, columns=columns)
    df['custom_key'] = df['patient_number'].apply(lambda item: int(item[1:]))
    df = df.sort_values(by='custom_key', ascending=True)
    df = df.drop(columns='custom_key')
    df.to_csv('../Tanveer/patient_num_and_metric.csv', index=False)
    
if __name__ == "__main__":
    main()
