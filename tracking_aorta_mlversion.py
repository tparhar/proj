import time

start_time = time.time()

import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv
import os
import glob
import pydicom
import shutil
import monai
import torch
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    LoadImage,
    EnsureChannelFirst,
    Flip,
    Rotate90,
    ScaleIntensityRangePercentiles
)
from monai.inferers import sliding_window_inference

from scipy.spatial import ConvexHull
from skimage.registration import optical_flow_tvl1 
from pdb import set_trace

"""
This program performs optical flow tracking on the descending thoracic aorta 
using MRI images of the patient's heart.

"""

path=r'new_patients/'
# all patient folders start with p
filenames = glob.glob(path + "p*")
for file in filenames: 
    filename=file
    # tells you what patient folder is being processed
    print(filename[12:])
    file_path = os.path.abspath(filename)
    file_list = os.listdir(file_path)
    dicom_files = [file for file in file_list]

    # Asserting that only 30 time steps are allowed
    dicom_files = dicom_files[:30]
    dicom_files = [file for file in file_list if file.split('.')[0].isdigit()]
    dicom_files.sort(key=lambda x: int(x.split('.')[0]))

    dicom_files_one = pydicom.dcmread(os.path.join(file_path, dicom_files[0]))
    image0 = dicom_files_one.pixel_array.astype(float)

    image0s = (np.maximum(image0, 0) / 500) * 255.0
    image0 = np.minimum(image0s, 255.0)
    image0 = np.uint8(image0)

    inference_transforms = Compose(
        [
            LoadImage(),
            EnsureChannelFirst(),
            ScaleIntensityRangePercentiles(lower=0, upper=100, b_min=0, b_max=1),
            # Rotate90 and Flip to fix orientation
            Rotate90(k=3, spatial_axes=(0, 1)),
            Flip(spatial_axis=1)
        ]
    )

    post_trans = Compose(
        [
            Activations(sigmoid=True),
            AsDiscrete(threshold=0.5)
        ]
    )

    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2
    )
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    # Need to define binaryImg at higher scope so it can be used later
    binaryImg = None
    with torch.no_grad():
        model_input = inference_transforms(os.path.join(file_path, dicom_files[0]))
        model_input = model_input.unsqueeze(0)
        model_output = sliding_window_inference(
            model_input,
            roi_size=[96, 96],
            sw_batch_size=4,
            predictor=model,
            overlap=0.25
        )
        model_output = post_trans(model_output)
        binaryImg = model_output[0][0].numpy()
    set_trace()
    # findContours only takes type uint8
    # cv2.RETR_CCOMP determines hierarchy, necessary but not used later
    # cv2.CHAIN_APPROX_NONE means that every single point of the contour is returned
    contours, hierarchy = cv2.findContours(binaryImg.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours:
        print("Bad segmentation, no mask produced.")
        destination_folder = os.path.join('bad_segmentations\\'+filename[12:])
        shutil.move(file_path, destination_folder)
        print(f"Moved folder '{filename}' to '{destination_folder}'")
        continue
    img = cv2.drawContours(binaryImg, contours, -1, (200,0,0), 2)
    set_trace()

    # The ConvexHull will be constructed from a set of (x, y) pairs
    conpoints = contours[0][:]
    mesh = []
    for i in range(len(conpoints)):
        # the columns for x and y are chosen arbitrarily, as long as they are kept
        # consistent, results should be accurate
        x0=conpoints[i][0,[1]][0]
        y0=conpoints[i][0,[0]][0]
        mesh.append([x0, y0])
    original_mesh = np.asarray(mesh)
    try:
        hull = ConvexHull(original_mesh)
    except:
        print("Bad segmentation, output was not contiguous")
        destination_folder = os.path.join('bad_segmentations\\'+filename[12:])
        shutil.move(file_path, destination_folder)
        print(f"Moved folder '{filename}' to '{destination_folder}'")
        continue

    fileout=file_path[:56]+'\\outfolders\\'+filename[12:]+'.csv'
    print(fileout)
    
    # This is the csv that the results of optical flow will be written to
    fileo = open(fileout, 'w+', newline ='')

    area_of_aorta = []
    print("Inital Area: {}".format(hull.area))
    area_of_aorta.append(hull.area)

    # OPTICAL FLOW

    # Saving ground truth that each subsequent image's displacement will be calculated from
    plt.imsave('temptime0.png',image0)
    image0 = cv2.imread('temptime0.png', cv2.IMREAD_GRAYSCALE)

    areas=[]
    area0=hull.area

    # takes image 1 through 29 (not including 0 because that was computed earlier)
    for j in range (1,len(dicom_files),1): 
        image1=  pydicom.dcmread(os.path.join(file_path, dicom_files[j]))
        image1 = image1.pixel_array.astype(float)

        # the max. part ensures that the array image0 only includes positive values 
        image1s = (np.maximum(image1, 0) / 500.0) * 255.0

        # the min. ensures that the array includes values that do not exceed 225
        image1 = np.minimum(image1s, 255.)

        # takes any whole number between 0-255, where in this case 0 represents black
        # (meaning no intensity) and 255 represents white (highest intensity) 
        image1 = np.uint8(image1)
        plt.imsave('temptime1.png',image1)
        image1 = cv2.imread('temptime1.png', cv2.IMREAD_GRAYSCALE)

        v1, u1 = optical_flow_tvl1(image0, image1)
        v2, u2 = optical_flow_tvl1(image1, image0)
        v=(v1-v2)/2 # average horizantal displacement
        u=(u1-u2)/2 # average vertical displacement
  
        # EXTRACTING THE MESH POINTS DISPLACEMENTS
        optiflow_time=1.
        meshtn=[]
   
        cx0=[];cy0=[];cx1=[];cy1=[];dx=[];dy=[]
        for i in range(len(conpoints)):
            x0=conpoints[i][0,[1]][0]
            y0=conpoints[i][0,[0]][0]

        #displacement
            meshtn.append([x0+v[x0,y0]*optiflow_time,y0+u[x0,y0]*optiflow_time]) #computes the displacement of the current point

        #adding coordinates/points to th empty lists initalized above
            cx0.append(x0)
            cy0.append(y0)
            cx1.append(x0+v[x0,y0]*optiflow_time)
            cy1.append(y0+u[x0,y0]*optiflow_time)
            dx.append(v[x0,y0]*optiflow_time)
            dy.append(u[x0,y0]*optiflow_time)

        
        meshn=np.asarray(meshtn)
        hull = ConvexHull(meshn)
        print(hull.area)
        area_of_aorta.append(hull.area) 

        if j == 1:
            areas.append([0,area0,cx0,cy0])
            areas.append([j,hull.area, cx1,cy1,dx,dy])
        else:
            areas.append([j,hull.area, cx1,cy1,dx,dy])

    with fileo:
        write = csv.writer(fileo)
        write.writerows(areas)
    
    destination_folder = os.path.join('used_folders\\'+filename[12:])
    shutil.move(file_path, destination_folder)
    print(f"Moved folder '{filename}' to '{destination_folder}'")

print("Finished program execution")
end_time = time.time()
elapsed_time = end_time - start_time
print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
