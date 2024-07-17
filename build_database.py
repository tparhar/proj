import os
import csv
import glob
import pydicom
import numpy as np
import cv2

from re import findall
from pdb import set_trace
from pathlib import Path
from ast import literal_eval

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
            ds.save_as(dicom_path)
            arr = patient_masks[idx].astype(ds.pixel_array.dtype)
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

def train_val_test_split(database, train_percent, val_percent, test_percent):
    sorted_patients = sorted(database.keys(), key=lambda item: int(item[1:]))
    set_trace() 
    num_patients = len(sorted_patients)
    slice_train_patients = int(train_percent * num_patients)
    slice_val_patients = int(val_percent * num_patients) + slice_train_patients

    train_patient_ids = sorted_patients[:slice_train_patients]
    val_patient_ids = sorted_patients[slice_train_patients: slice_val_patients]
    test_patient_ids = sorted_patients[slice_val_patients:]

    train_patients = list()
    val_patients = list()
    test_patients = list()

    for patient_id in train_patient_ids:
        list_of_dicts = database[patient_id]
        for dict in list_of_dicts:
            train_patients.append(dict)

    for patient_id in val_patient_ids:
        list_of_dicts = database[patient_id]
        for dict in list_of_dicts:
            val_patients.append(dict)

    for patient_id in test_patient_ids:
        list_of_dicts = database[patient_id]
        for dict in list_of_dicts:
            test_patients.append(dict)
    
    return train_patients, val_patients, test_patients
    



if __name__ == "__main__":
    # TESTING
    dataset = {
    "p10": [{"img": 5}, {"img": 6}],
    "p2": [{"img": 3}, {"img": 4}],
    "p306": [{"img": 9}, {"img": 10}],
    "p25": [{"img": 7}, {"img": 8}],
    "p1": [{"img": 1}, {"img": 2}]
    }
    train_patients, val_patients, test_patients = train_val_test_split(dataset, 0.5, 0.3, 0.2)