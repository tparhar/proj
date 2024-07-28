# Code taken from here:
# https://github.com/Project-MONAI/tutorials/blob/main/2d_segmentation/torch/unet_training_dict.py
# Modified the input data to suit my project

import os
import logging

import torch
import torchvision
import wandb
import yaml

import monai
from monai.data import pad_list_data_collate, decollate_batch, DataLoader
from monai.metrics import ConfusionMatrixMetric, DiceMetric, MeanIoU
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    LoadImaged,
    RandAdjustContrastd,
    RandAxisFlipd,
    RandRotate90d,
    Rand2DElasticd,
    Resized,
    ScaleIntensityRangePercentilesd,
)
from monai.visualize import plot_2d_or_3d_image

import build_database
from pdb import set_trace

def main():
    with open("./yaml_files/testing.yaml") as file:
        config = yaml.safe_load(file)
    
    run = wandb.init(
        project="my-awesome-project",
        config=config,
        tags=["Testing"],
        notes="Testing if the new transforms even work without throwing an error"
    )

    database_folder = 'toy_rand'

    num_epochs = wandb.config.epochs
    lr = wandb.config.lr
    train_batch_size = wandb.config.train_batch_size
    val_batch_size = wandb.config.val_batch_size
    num_workers = wandb.config.num_workers

    print(
        "HYPERPARAMETERS\n"\
        "---------------\n"\
        "Epochs: {}\n"\
        "Learning Rate: {}\n"\
        "Train Batch Size: {}\n"\
        "Validation Batch Size: {}\n"\
        "Num Workers: {}\n".format(num_epochs, lr, train_batch_size, val_batch_size, num_workers)
    )
    logger = logging.getLogger('pydicom')
    logger.disabled = True

    dataset = build_database.parse_database(database_folder)
    train_files, val_files, test_files = build_database.split_data(dataset, train_percent=0.7, val_percent=0.3, test_percent=0.0)
    # define transforms for image and segmentation
    train_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            Resized(keys=["img", "seg"], spatial_size=[256, 256], mode=["bilinear", "nearest"]),
            ScaleIntensityRangePercentilesd(keys="img", lower=0, upper=100, b_min=0, b_max=1),
            RandAdjustContrastd(keys=["img"], prob=0.1, gamma=(0.5, 4.5)),
            Rand2DElasticd(keys=["img", "seg"], prob=0.1, spacing=(20, 20), magnitude_range=(1, 2), padding_mode="reflection", mode=["bilinear", "nearest"]),
            RandAxisFlipd(keys=["img", "seg"], prob=0.5),
            RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=(0, 1)),
            ScaleIntensityRangePercentilesd(keys="img", lower=5, upper=95, b_min=0, b_max=1),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            Resized(keys=["img", "seg"], spatial_size=[256, 256], mode=["bilinear", "nearest"]),
            ScaleIntensityRangePercentilesd(keys=["img"], lower=0, upper=100, b_min=0, b_max=1),
        ]
    )

    post_trans = Compose(
        [
            Activations(sigmoid=True),
            AsDiscrete(threshold=0.5)
        ]
    )

    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(
        train_ds,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=pad_list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )

    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=val_batch_size, num_workers=num_workers, collate_fn=pad_list_data_collate)

    dice_metric = DiceMetric()
    confusion_matrix_metrics_function = ConfusionMatrixMetric(metric_name=["precision", "recall", "f1 score"], compute_sample=True)
    iou_metric_function = MeanIoU()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    loss_function = monai.losses.DiceCELoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # start a typical PyTorch training
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1

    # saving model, and logging it as artifact to wandb
    model_save_path = os.path.join(wandb.run.dir, 'model.pth')

    # Training Loop
    for epoch in range(num_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{num_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["img"].to(device), batch_data["seg"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= step
        wandb.log({"train_loss": epoch_loss})
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                for val_data in val_loader:
                    val_images, val_labels = val_data["img"].to(device), val_data["seg"].to(device)
                    val_outputs = model(val_images)
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    val_outputs = torch.stack(val_outputs)
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)
                    confusion_matrix_metrics_function(y_pred=val_outputs, y=val_labels)
                    iou_metric_function(y_pred=val_outputs, y=val_labels)
            
                # aggregate the final mean metrics result
                metric = dice_metric.aggregate().item()
                precision_metric = confusion_matrix_metrics_function.aggregate()[0].item()
                recall_metric = confusion_matrix_metrics_function.aggregate()[1].item()
                f1_score_metric = confusion_matrix_metrics_function.aggregate()[2].item()
                iou_metric = iou_metric_function.aggregate().item()

                val_image = val_images[0].permute(1, 2, 0).cpu().numpy()
                val_label = val_labels[0][0].cpu().numpy()
                val_output = val_outputs[0][0].cpu().numpy()

                mask_imgs = wandb.Image(
                    val_image,
                    masks={
                        "predictions": {"mask_data": val_output},
                        "ground_truth": {"mask_data": val_label}
                    }
                )


                wandb.log(
                    {
                        "dice": metric,
                        "precision": precision_metric,
                        "recall": recall_metric,
                        "f1 score": f1_score_metric,
                        "iou": iou_metric,
                        "examples": mask_imgs
                    }
                )

                # reset the status for next validation round
                dice_metric.reset()
                confusion_matrix_metrics_function.reset()
                iou_metric_function.reset()

                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), model_save_path)
                    print("saved new best metric model")

                # printing all metrics to console
                print(
                    "current epoch: {} current mean dice: {:.4f} current mean precision: {:.4f}\n"\
                    "current mean recall: {:.4f} current mean f1 score: {:.4f} current mean iou: {:.4f}\n"\
                    "best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric, precision_metric, recall_metric, f1_score_metric, iou_metric, best_metric, best_metric_epoch
                    )
                )

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")

if __name__ == "__main__":
    main()