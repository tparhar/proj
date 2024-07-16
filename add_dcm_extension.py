import os
import glob
import pydicom

from pdb import set_trace

path = r"new_patients"

patients = glob.glob('p*', root_dir=path)

for patient in patients:
    files = [file for file in os.listdir(os.path.join(path, patient)) if ((not file.endswith(".dcm")) and (not file.endswith(".npy")))]
    for file in files:
        file_path = os.path.join(path, patient, file)
        try:
            pydicom.dcmread(file_path)
            os.rename(file_path, file_path + ".dcm")
        except:
            pass