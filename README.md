# Tanveer's Summer 2024 Research

All of the work that I did this summer, including:
1. The Python script for training a U-Net neural network that performs segmentation on the descending thoracic aorta in top-down DICOM images of patients' hearts.

2. The fully trained U-Net model.

3. The strain tracking script that takes patients' DICOM images, uses the trained model, and outputs csv files detailing their strain through frames in the cardiac cycle.

Patients need to be sorted into a couple different folders. This script expects DICOM images, with a folder called `new_patients/` containing each patient as a folder labelled in the format `p<number>/`. Inside each folder is the patient's dicom images, labelled from `0.dcm` to `29.dcm`.

To generate the masks associated with each image, the script expects a `csv_files/` folder that contains each patient's csv with a set of coordinates associated with each DICOM image, labelled `p<num>-<some_id_number>.csv`.

Using utilities in `build_database.py`, namely the `build_database()` function, you will need to create a folder titled `imgs_and_segs`, which will contain every single DICOM image and its associated mask. You will only need to run `build_database()` once to populate the directory, from there the scripts will handle the rest.

## Model Training Script

The main training file is labelled `monaitrain_wandb.py`, and can be run by following the instructions below.



### To Run Training Script

There are 3 config yaml files available to run, `baseline.yaml`, `onlyzeros.yaml`, and `testing.yaml`. You should use `onlyzeros.yaml` when training as it will train much faster and the data will have much more variety. `testing.yaml` should only be used when debugging as it takes mere minutes to run, and it should be run with a toy dataset that contains a slice of your full dataset. It should also not be run as a `wandb sweep` and instead through the usual `python monaitrain_wandb.py`. `baseline.yaml` can be used if you want all 30 images from each patient that you run on - be warned, this will increase the training time immensely, and may not significantly improve or even reduce model accuracy. 

```
# Setting up Docker Environment
docker run -it --rm -e WANDB_API_KEY=Tanveers_wandb_key dockerfile_name

# THIS WILL PRINT OUT A SWEEP ID THAT YOU USE IN THE NEXT LINE
wandb sweep --config=./yaml_files/onlyzeros.yaml
# USE THAT SWEEP ID HERE TO RUN THE SWEEP
wandb agent <your_sweep_id>
```

### To Run Training Script for Debugging
```
# Assuming you've already done
# docker run -it --rm WANDB_API_KEY=Tanveers_wandb_key dockerfile_name
python monaitrain_wandb.py --config=./yaml_files/testing.yaml
```

## Strain Tracking Script
This strain tracking script requires you to follow the instructions in the first section to set up the folder structure, along with the setup of the docker environment as shown in the **To Run Training Script** section. After that is done, simply run the following line of code:
```
python tracking_aorta_mlversion.py
```

This code will take the patients in `new_patients/`, use the model `model.pth` and generate csv files into `csv_files`for each patient containing data such as max area, strain, and the coordinates at each image interval.
