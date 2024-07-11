import numpy as np
import pandas as pd
from ast import literal_eval
from torch.utils.data import Dataset
import glob
import os
import pydicom
import cv2
from pydicom.pixel_data_handlers.util import apply_voi_lut
import csv
import re
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as v2
import torch
import torchvision.transforms.functional as F
from pdb import set_trace

# Define Functions for the Rest of the Program
#Loading a Single Dicom Image
def load_dicom_image(file_path: str) -> np.ndarray: #the dicom image being loaded is "MONOCHROME2", which means 
    dicom = pydicom.dcmread(file_path) #parses the dicom image, returns a dicom dataset
    image = apply_voi_lut(dicom.pixel_array, dicom) #turns the dicom dataset into an ndarray to pass into this argument, returns an array of type np.float64, shape 192x192
    image = image - np.min(image) #these next two lines rescale the image to be between 0 and 1
    image = image / np.max(image)
    return image

def show_dicom_image(image):
    plt.imshow(image, cmap= 'gray')

def load_dicom_images(folder_path: str) -> np.ndarray: #make sure this is abs_path to directory
    # Sorting the input directory
    image_nums = os.listdir(folder_path)
    image_nums.sort(key=lambda a: int(a))

    first_image_path = os.path.join(folder_path, image_nums[0])
    first_image = load_dicom_image(first_image_path)

    image_shape = first_image.shape
    num_images = len(image_nums)

    all_images = np.empty((num_images, *image_shape), dtype=first_image.dtype)

    index=0

    for image in image_nums:
        dicom_image = load_dicom_image(os.path.join(folder_path, image))
        all_images[index] = dicom_image
        index += 1
    return all_images

def load_images_and_masks(folder_path: str) -> list[np.ndarray]:
    patients = glob.glob(os.path.join(folder_path, 'p*'))
    all_patient_images = list()
    all_patient_masks = list()
    pairs = csv_extractor()
    for idx, patient in enumerate(patients):
        single_patient_images = load_dicom_images(os.path.join(patient))
        image_shape = single_patient_images.shape[1:]
        all_patient_masks.append(create_masks_from_convex_hull(image_shape, pairs[idx]))
        all_patient_images.append(single_patient_images)
    return all_patient_images, all_patient_masks

def create_mask_from_convex_hull(image_shape, convex_hull_coords):
    mask = np.zeros(image_shape, dtype=np.float32)
    convex_hull_coords = np.array(convex_hull_coords, dtype=np.int32)
    cv2.fillConvexPoly(mask, convex_hull_coords, 1)
    return mask

def create_masks_from_convex_hull(image_shape: tuple, convex_hull_coords: np.ndarray):
    num_masks = convex_hull_coords.shape[0]
    all_masks = np.empty((num_masks, *image_shape), dtype=np.float32)
    index = 0
    for i in range(num_masks):
        all_masks[index] = create_mask_from_convex_hull(image_shape, convex_hull_coords[i])
        index += 1
    return all_masks

#Parsing CSV of patients to retrieve mask coordinates
def csv_extractor() -> list[np.ndarray]:
    path=r'csv_files/'

    filenames = sorted( filter( os.path.isfile,
                            glob.glob(path + '*.csv') ) )
    patients=[]
    for file in filenames:
        print(file)
        num=file[11:14]
        num1=re.findall('[0-9]+',num)
        num2=int(num1[0])
        print(num,num2)
        x=[]; y=[]
        #obtaining coordinates
        with open(file, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                x.append(row[3])
                y.append(row[2])
        def commas_to_ndarray_csv(x):
            new_list=[]
            for i in range(len(x)):
                new_list_segment = literal_eval(x[i])
                new_list.append(new_list_segment)
            new_list_arr = np.array(new_list)
            return new_list_arr

        x_coords = commas_to_ndarray_csv(x)
        y_coords = commas_to_ndarray_csv(y)
        def coords_to_pairs(x, y):
            # Using list comprehension to combine x and y into tuples
            combined = np.array([[(x_val, y_val) for x_val, y_val in zip(x_row, y_row)] for x_row, y_row in zip(x, y)])
            return combined

        pairs = coords_to_pairs(x_coords, y_coords)
        patients.append(pairs)
    return patients


if __name__ == "__main__":
    #Getting DICOM Images
    #Using Patient 301

    p301_all_dicom_images = load_dicom_images(r'C:/Users/mrtan/Desktop/Job Stuff/Sum2024Research/proj/newpatients/p301/')
    p301_image_shape = p301_all_dicom_images[0].shape
    print(p301_image_shape)
    plt.imshow(p301_all_dicom_images[0], cmap="gray")
    plt.show()



    # Creating Masks from the Pairs
    # Using Patient 301
    p301_mask_coords = np.load('unsorted/p301_mask_coords.npy')
    print(p301_mask_coords.shape)

    p301_all_masks = create_masks_from_convex_hull(p301_image_shape, p301_mask_coords)
    print(p301_all_masks.shape)
    print(p301_all_masks.dtype)

    plt.imshow(p301_all_masks[0], cmap="gray")
    plt.show()

    #Testing Data (choosing 5 from the end)
    p301_test_images = p301_all_dicom_images[25:]
    p301_test_masks = p301_all_masks[25:]

    print(p301_test_images.shape)
    print(p301_test_masks.shape)
    print(type(p301_test_images))
    print(type(p301_test_masks))

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(p301_test_images[0], cmap="gray")
    ax[1].imshow(p301_test_masks[0], cmap="gray")
    plt.show()

    np.save('test_images/p301_test_images.npy', p301_test_images)
    np.save('test_masks/p301_test_masks.npy', p301_test_masks)

    #Training Data
    p301_train_images = p301_all_dicom_images[:20]
    p301_train_masks = p301_all_masks[:20]

    print(p301_train_images.shape)
    print(p301_train_masks.shape)
    print(type(p301_train_images))
    print(type(p301_train_masks))

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(p301_train_images[0], cmap="gray")
    ax[1].imshow(p301_train_masks[0], cmap="gray")
    plt.show()

    np.save('train_images/p301_train_images.npy', p301_train_images)
    np.save('train_masks/p301_train_masks.npy', p301_train_masks)

    #Validating Data
    p301_val_images = p301_all_dicom_images[20:25]
    p301_val_masks = p301_all_masks[20:25]

    print(p301_val_images.shape)
    print(p301_val_masks.shape)
    print(type(p301_val_images))
    print(type(p301_val_masks))

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(p301_val_images[0], cmap="gray")
    ax[1].imshow(p301_val_masks[0], cmap="gray")
    plt.show()

    np.save('val_images/p301_val_images.npy', p301_val_images)
    np.save('val_masks/p301_val_masks.npy', p301_val_masks)