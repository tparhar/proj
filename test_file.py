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

import prepdata

masks: list[list[np.ndarray]] = prepdata.load_masks('new_patients')
dicom_patient_paths: list[list[str]] = prepdata.list_image_paths('new_patients')
mask_paths = prepdata.np_masks_to_dcm(masks, dicom_patient_paths, 'rand')
set_trace()
image_paths = prepdata.create_tempdir_paths('new_patients', 'rand')