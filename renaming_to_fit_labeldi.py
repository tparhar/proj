import os
import glob

from pdb import set_trace

patients = glob.glob('p*', root_dir='healthy_patients')
patient_csvs = glob.glob('*.csv', root_dir='healthy_patient_csvs')
starting_num = 311
for patient in patients:
    patient_full_path = os.path.join('healthy_patients', patient)
    dicoms = sorted(os.listdir(patient_full_path))
    for idx, dicom in enumerate(dicoms):
        dicom_full_path = os.path.join(patient_full_path, dicom)
        dicom_new_path = os.path.join(patient_full_path, str(idx) + ".dcm")
        os.rename(dicom_full_path, dicom_new_path)
    patient_id_after_dash = patient.split('-', 1)[1]
    patient_new_name = 'p' + str(starting_num) + '-' + patient_id_after_dash
    patient_new_path = os.path.join('healthy_patients', patient_new_name)
    patient_csv = glob.glob('*' + patient_id_after_dash + "*", root_dir='healthy_patient_csvs')
    if patient_csv is None:
        raise ValueError("This csv doesn't exist")
    patient_csv = patient_csv[0]
    patient_csv_path = os.path.join('healthy_patient_csvs', patient_csv)
    patient_new_csv_path = os.path.join('healthy_patient_csvs', patient_new_name + ".csv")
    os.rename(patient_full_path, patient_new_path)
    os.rename(patient_csv_path, patient_new_csv_path)
    starting_num += 1