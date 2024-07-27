import os
import csv
import glob
import json
import random
import pydicom
import numpy as np
import cv2

from re import findall
from pdb import set_trace
from pathlib import Path
from ast import literal_eval
from collections import defaultdict

import pydicom.encaps

#Parsing CSV of patients to retrieve mask coordinates
def csv_extractor(csv_folder, patient_id) -> list[np.ndarray]:
    match = glob.glob(patient_id + "*", root_dir=csv_folder)

    # CHECKS IF THERE IS A MATCH OR NOT
    if not match:
        return None
    file = os.path.join(csv_folder, match[0])

    try:
        with open(file, newline='') as f:
            reader = csv.reader(f)
    except:
        return None
    
    x=[]; y=[]
    #obtaining coordinates
    with open(file, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            x.append(row[3])
            y.append(row[2])

    def commas_to_list_of_lists(x):
        new_list=[]
        for i in range(len(x)):
            new_list_segment = literal_eval(x[i])
            new_list.append(new_list_segment)
        return new_list

    x_coords = commas_to_list_of_lists(x)
    y_coords = commas_to_list_of_lists(y)

    # Using list comprehension to combine x and y into tuples
    pairs = list()
    for x_row, y_row in zip(x_coords, y_coords):
        coords_list = list()
        for x_val, y_val in zip(x_row, y_row):
            coords_list.append((x_val, y_val))
        pairs.append(coords_list)

    return pairs

def create_masks(image_shape, coord_pairs):
    num_masks = len(coord_pairs)
    all_masks = list()
    for i in range(num_masks):
        mask = np.zeros(image_shape, dtype=np.float32)
        convex_hull_coords = np.array(coord_pairs[i], dtype=np.int32)
        cv2.fillConvexPoly(mask, convex_hull_coords, 1)
        all_masks.append(mask)
    return all_masks


def build_database(patients_folder, csv_folder, outdir):
    patients = glob.glob('p*', root_dir=patients_folder)
    img_and_seg_dict = dict()
    for patient in patients:
        patient_num = patient.split('-')[0]
        print(patient_num)
        # GET THE DICOM FILE PATHS FOR PATIENT -------------------------------- 
        patient_folder_path = os.path.join(patients_folder, patient)
        dicom_file_paths = [os.path.join(patient_folder_path, file) for file in os.listdir(patient_folder_path) if (file.endswith(".dcm") and (not os.path.basename(file) == "AORTA_DESC.dcm"))]
        dicom_file_paths = sorted(dicom_file_paths, key=lambda a: int(Path(a).stem))[:30] # limiting them to only 30 images

        # GET THE COORDINATE PAIRS AND GENERATE MASKS FOR PATIENT
        coord_pairs = csv_extractor(csv_folder, patient)
        if coord_pairs == None:
            print("This csv file doesn't exist")
            continue

        first_image_shape = pydicom.dcmread(dicom_file_paths[0]).pixel_array.shape
        patient_masks = create_masks(first_image_shape, coord_pairs)

        img_and_seg_list = list()
        img_and_seg_dict[patient_num] = img_and_seg_list

        for idx, val in enumerate(dicom_file_paths):
            dicom_path = os.path.join(outdir, patient_num + "_img_" + os.path.basename(val))
            ds = pydicom.dcmread(dicom_file_paths[idx])
            try:
                arr = patient_masks[idx].astype(ds.pixel_array.dtype)
            except:
                print("The csv file was missing time steps")
                break

            ds.save_as(dicom_path)
            ds.PixelData = arr.tobytes()

            ds.Rows, ds.Columns = arr.shape
            ds.LargestImagePixelValue = int(arr.max())
            ds.SmallestImagePixelValue = int(arr.min())
            ds.BitsAllocated = arr.itemsize * 8
            ds.BitsStored = arr.itemsize * 8
            ds.HighBit = ds.BitsStored - 1
            ds.PixelRepresentation = 0

            ds.Rows = int(ds.Rows)
            ds.Columns = int(ds.Columns)
            ds.BitsAllocated = int(ds.BitsAllocated)
            ds.BitsStored = int(ds.BitsStored)
            ds.HighBit = int(ds.HighBit)
            ds.PixelRepresentation = int(ds.PixelRepresentation)

            ds.WindowCenter = int((arr.max() + arr.min()) / 2)
            ds.WindowWidth = int(arr.max() - arr.min())

            mask_path = os.path.join(outdir, patient_num + "_seg_" + os.path.basename(val))
            ds.save_as(mask_path)
            img_and_seg_list.append({"img": dicom_path, "seg": mask_path})
    return img_and_seg_dict

def parse_database(database_folder):
    # Directory containing the files
    # Initialize the dictionary
    sorted_dict = defaultdict(list)

    # Process each file in the directory
    for file in os.listdir(database_folder):
        # Check if the file has the correct extension
        if file.endswith(".dcm"):
            # Split the filename into parts
            parts = file.split('_')
            patient = parts[0]  # e.g., 'p0', 'p1'
            file_type = parts[1]  # e.g., 'img', 'seg'
            index = parts[2].split('.')[0]  # e.g., '0', '1'

            # Find or create the appropriate entry in the dictionary
            if len(sorted_dict[patient]) <= int(index):
                # Extend the list to accommodate the new index
                sorted_dict[patient].extend([{}] * (int(index) + 1 - len(sorted_dict[patient])))

            # Add the file with its relative path to the appropriate dictionary within the list
            relative_path = os.path.join(database_folder, file)
            sorted_dict[patient][int(index)][file_type] = relative_path

    # Convert defaultdict to regular dict for output
    sorted_dict = dict(sorted_dict)
    
    return sorted_dict

import os
from collections import defaultdict

def split_data(data_dict, train_percent=0.7, val_percent=0.15, test_percent=0.15):
    # Ensure the percentages sum to 1
    assert train_percent + val_percent + test_percent == 1, "Percentages must sum to 1."

    # Sort the patient keys
    patients = sorted(data_dict.keys(), key=lambda item: int(item[1:]))
    set_trace()
    random.Random(42).shuffle(patients)

    # Calculate the number of patients for each set
    total_patients = len(patients)
    train_count = int(total_patients * train_percent)
    val_count = int(total_patients * val_percent)
    test_count = total_patients - train_count - val_count

    # Split the patients
    train_patients = patients[:train_count]
    val_patients = patients[train_count:train_count + val_count]
    test_patients = patients[train_count + val_count:]

    # Initialize the result lists
    train_set = []
    val_set = []
    test_set = []

    def is_valid_pair(pair):
        img = pair.get('img', '')
        seg = pair.get('seg', '')
        return img and seg and img.endswith('img_0.dcm') and seg.endswith('seg_0.dcm')

    for patient in train_patients:
        train_set.extend([pair for pair in data_dict[patient] if pair.get('img', '') and pair.get('seg', '')])

    # Assign data to the validation and test sets
    for patient in val_patients:
        val_set.extend([pair for pair in data_dict[patient] if is_valid_pair(pair)])

    for patient in test_patients:
        test_set.extend([pair for pair in data_dict[patient] if is_valid_pair(pair)])

    return train_set, val_set, test_set

def load_hyperparameters(file_path, *args):
    with open(file_path, 'r') as file:
        all_params = json.load(file)
    
    base_params = all_params.get("base", {})
    for param_set in args:
        base_params.update(all_params.get(param_set, {}))
    
    return base_params

if __name__ == "__main__":
    # TESTING
    # build_database('new_patients', 'csv_files', 'imgs_and_segs')
    sorted_dictionary = parse_database('imgs_and_segs')
    train, val, test = split_data(sorted_dictionary)