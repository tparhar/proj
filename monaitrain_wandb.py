# Code taken from here:
# https://github.com/Project-MONAI/tutorials/blob/main/2d_segmentation/torch/unet_training_dict.py
# Modified the input data to suit my project

import os
import logging

import torch
import torchvision
import wandb
import yaml
import argparse

import monai
from monai.data import pad_list_data_collate, decollate_batch, DataLoader
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
    parser = argparse.ArgumentParser(description="Run a WandB sweep with a specified config file.")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML config file.")

    args, unknown = parser.parse_known_args()

    config = build_database.load_config(args.config)

    set_trace()

    if 'baseline' in args.config:
        run = wandb.init(
            project="proj",
            config=config,
            tags=["Baseline"],
            notes="Testing how baseline performs"
        )

        database_folder = 'imgs_and_segs'

        num_epochs = wandb.config.epochs
        lr = wandb.config.lr
        train_batch_size = wandb.config.train_batch_size
        val_batch_size = wandb.config.val_batch_size
        num_workers = wandb.config.num_workers
        patch_size = None
        spatial_crop_num_samples = None
        overlap = None

        transform_select = 'baseline'
        inference_select = 'regular'

        dataset = build_database.parse_database(database_folder)
        train_files, val_files, test_files = build_database.split_data(dataset, onlyzeros=False, train_percent=0.7, val_percent=0.3, test_percent=0.0)
    elif 'onlyzeros' in args.config:
        run = wandb.init(
            project="proj",
            config=config,
            tags=["OnlyZeros"],
            notes="Checking performance of onlyzeros config"
        )

        database_folder = 'imgs_and_segs'

        num_epochs = wandb.config.epochs
        lr = wandb.config.lr
        train_batch_size = wandb.config.train_batch_size
        val_batch_size = wandb.config.val_batch_size
        num_workers = wandb.config.num_workers
        patch_size = wandb.config.spatial_crop
        spatial_crop_num_samples = wandb.config.spatial_crop_num_samples
        overlap = wandb.config.overlap

        transform_select = 'onlyzeros'
        inference_select = 'sliding_window'

        dataset = build_database.parse_database(database_folder)
        train_files, val_files, test_files = build_database.split_data(dataset, onlyzeros=True, train_percent=0.7, val_percent=0.3, test_percent=0.0)
    elif 'testing' in args.config:
        run = wandb.init(
            project="my-awesome-project",
            config=config,
            tags=["Testing"],
            notes="Regular testing - no special config"
        )

        database_folder = 'toy_rand'

        num_epochs = wandb.config.epochs
        lr = wandb.config.lr
        train_batch_size = wandb.config.train_batch_size
        val_batch_size = wandb.config.val_batch_size
        num_workers = wandb.config.num_workers
        patch_size = None
        spatial_crop_num_samples = None
        overlap = None

        transform_select = 'testing'
        inference_select = 'regular'

        dataset = build_database.parse_database(database_folder)
        train_files, val_files, test_files = build_database.split_data(dataset, onlyzeros=True, train_percent=0.7, val_percent=0.3, test_percent=0.0)
    else:
        raise ValueError("Some random error, don't know how it happened.")

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

    train_transforms = build_database.TransformDict(transform_select, patch_size, spatial_crop_num_samples).return_transform_dict()["training"]
    val_transforms = build_database.TransformDict(transform_select, patch_size, spatial_crop_num_samples).return_transform_dict()["validation"]
    test_transforms = build_database.TransformDict(transform_select, patch_size, spatial_crop_num_samples).return_transform_dict()["test"]

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
    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size,
        num_workers=num_workers,
        collate_fn=pad_list_data_collate
    )

    test_ds = monai.data.Dataset(data=test_files, transform=test_transforms)
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        num_workers=num_workers,
        collate_fn=pad_list_data_collate
    )

    dice_metric = DiceMetric(include_background=False)
    confusion_matrix_metrics_function = ConfusionMatrixMetric(include_background=False, metric_name=["precision", "recall", "f1 score"], compute_sample=True)
    iou_metric_function = MeanIoU(include_background=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    loss_function = monai.losses.DiceCELoss(include_background=False, sigmoid=True)
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
                    if inference_select == 'sliding_window':
                        val_outputs = sliding_window_inference(
                            val_images,
                            patch_size,
                            sw_batch_size=spatial_crop_num_samples,
                            predictor=model,
                            overlap=overlap
                        )
                    else:
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