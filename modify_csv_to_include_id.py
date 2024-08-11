import pandas as pd
from pdb import set_trace
import glob
import os

patient_num_and_metric_df = pd.read_csv('patient_num_and_metric.csv')
patient_ids = list()

for patient_num in patient_num_and_metric_df['patient_number']:
    found_id = glob.glob(patient_num + '*', root_dir='new_patients')[0]
    patient_ids.append(found_id)

patient_num_and_metric_df['patient_id'] = patient_ids

patient_id_and_metric_df = patient_num_and_metric_df.drop(columns='patient_number')
patient_id_and_metric_df = patient_id_and_metric_df[['patient_id', 'dice']]
patient_id_and_metric_df.to_csv('patient_id_and_metric.csv', index=False)
