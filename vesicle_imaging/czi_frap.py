import matplotlib.pyplot as plt
from matplotlib import pylab as pl
import matplotlib as mpl
import numpy as np
import scipy.special
import scipy.optimize
import czifile
from skimage import io, color
from skimage.transform import rescale, resize, downscale_local_mean
import pandas as pd
import importlib
from vesicle_imaging import imgfileutils as imf
from vesicle_imaging import czi_image_handling as handler
from vesicle_imaging import czi_image_analysis as analysis
from vesicle_imaging import czi_zstack_analysis as zst
from vesicle_imaging import czi_timelapse_analysis as tma
from vesicle_imaging import format
from math import pi
import multiprocessing
from itertools import chain # -> list(chain(*fluorescein_image_pre))
import cv2
import warnings
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import os
from collections import OrderedDict
from matplotlib.widgets import Slider

from icecream import ic

path = '/Users/heuberger/code/vesicle-imaging/test_data'
os.listdir(path)
curr_dir = os.getcwd()
os.chdir(path)

files, filenames = handler.get_files(path)
filenames.sort(reverse=True)
files.sort(reverse=True)
print (files)

#print (images)
print ('number of images: ', len(files))

print (files)
#filenames = files
#filenames.sort()
for i in range(0,len(files)):
    filenames[i] = files[i].replace('.czi','')
ic(filenames)


image_data = []
image_metadata = []
image_add_metadata = []
for image in files:
    print (image)
    data, metadata, add_metadata = handler.load_image_data([image])
    image_data.append(data)
    image_metadata.append(metadata)
    image_add_metadata.append(add_metadata)
print ('IMAGE IMPORT DONE!')
os.chdir(curr_dir)

ic(image_data[0][0].shape)

image_cxyz_data = handler.extract_channels_timelapse([image_data[0][0]])
#image_xy_data[0][0].shape
ic(image_cxyz_data[0].shape)

handler.disp_basic_img_info(image_cxyz_data, image_metadata)


#handler.disp_all_metadata(image_metadata)

# to save images as pngs
saving_on = False
if saving_on == True:
    try:
        os.mkdir('FRAP_analysis')#todo make that this goes to the location of the input file
    except FileExistsError:
        pass

channels = ['rhoPE', 'Transmission']
ic(image_cxyz_data[0][:,0,:,:].shape)

fig = plt.figure()
plt.imshow(image_cxyz_data[0][:,5,:,:][0])
plt.show()

# for frame in range(0,10):
#     plt.imshow(image_cxyz_data[0][:,frame,:,:][0], cmap='hot')
#     plt.show()

ic(image_add_metadata[0][0]['Layers']['Layer'][0]['Elements']['Circle']['Geometry'])
frap_position=image_add_metadata[0][0]['Layers']['Layer'][0]['Elements']['Circle']['Geometry'] #centerx, centery, radius
ic(frap_position)
#ic(frap_position['CenterX'])

frap_position = dict(frap_position)
ic(frap_position)

ic(int(float(frap_position['CenterX'])))

fig = plt.figure()
output = image_cxyz_data[0][:,5,:,:][0].copy()
cv2.circle(output, (int(float(frap_position['CenterX'])), int(float(frap_position['CenterY']))), int(float(frap_position['Radius'])), (255, 255, 255), 2)  # x,y,radius
#cv2.circle(output, (int(float(frap_position['CenterX']))+30, int(float(frap_position['CenterY']))+30), int(float(frap_position['Radius'])), (0, 0, 0), 2)  # x,y,radius
plt.imshow(output)
plt.colorbar()
plt.show()

num_frames = len(image_cxyz_data[0][:,:,:,:][0])
bleach_frame = 5

stats = []

x_0 = int(float(frap_position['CenterX']))
y_0 = int(float(frap_position['CenterY']))
radius = int(float(frap_position['Radius']))
ic(x_0, y_0, radius)

for frame in range(0, num_frames):
    #ic(frame)
    measurement_image = image_cxyz_data[0][:,frame,:,:][0].copy()

    #blank_image = np.zeros((256,256,3), np.uint8)

    # make radius slighly smaller so border is not in range
    distance_from_border = 1
    measurement_radius = radius - distance_from_border  # [index]
    #ic(measurement_radius)

    pixels_in_circle = []

    for x in range(x_0 - measurement_radius, x_0 + measurement_radius):
        for y in range(y_0 - measurement_radius, y_0 + measurement_radius):
            dx = x - x_0
            dy = y - y_0

            distanceSquared = dx ** 2 + dy ** 2

            if distanceSquared <= (measurement_radius ** 2):
                pixel_val = measurement_image[y][x]
                pixels_in_circle.append(pixel_val)
                #cv2.circle(blank_image, (x, y), radius=0, color=(255, 0, 255), thickness=-1)
    # plt.figure()
    # plt.imshow(blank_image)
    # plt.show()

    #print('filename: ', filenames[index])
    #print('no. of GUVs counted: ', len(detected_circles[index]))
    #print('number of pixels: ', len(pixels_in_circle))
    print('min: ', np.min(pixels_in_circle))
    print('max: ', np.max(pixels_in_circle))
    print('average: ', np.mean(pixels_in_circle))
    print('stdev: ', np.std(pixels_in_circle))
    stats.append(np.mean(pixels_in_circle))

#ic(stats)
plt.figure()
plt.plot(stats)
plt.show()

# Compute I_0
I_0 = 0.0
I_pre = np.empty(bleach_frame)
ic(I_pre)
for i in range(bleach_frame):
    measurement_image = image_cxyz_data[0][:, i, :, :][0]

    x_0 = int(float(frap_position['CenterX']))
    y_0 = int(float(frap_position['CenterY']))
    radius = int(float(frap_position['Radius']))
    distance_from_border = 1
    # make radius slighly smaller so border is not in range
    measurement_radius = radius - distance_from_border  # [index]
    #ic(measurement_radius)

    pixels_in_circle = []

    for x in range(x_0 - measurement_radius, x_0 + measurement_radius):
        for y in range(y_0 - measurement_radius, y_0 + measurement_radius):
            dx = x - int(float(frap_position['CenterX']))
            dy = y - int(float(frap_position['CenterY']))
            distanceSquared = dx ** 2 + dy ** 2
            # print (distanceSquared)
            if distanceSquared <= (measurement_radius ** 2):
                pixel_val = measurement_image[y][x]
                pixels_in_circle.append(pixel_val)
    I_pre[i] = np.mean(pixels_in_circle)

I_0 = I_pre.sum() / bleach_frame
ic(I_0)

# Reset time to that time = 0 is on bleach frame
t_pre = range(0,5)#frap_xyt.t[:bleach_frame] - frap_xyt.t[bleach_frame]
t = range(5,num_frames)#rap_xyt.t[bleach_frame:] - frap_xyt.t[bleach_frame]

# Compute average I over time
I_mean = np.empty(num_frames - bleach_frame)
for i in range(bleach_frame, num_frames):
    measurement_image = image_cxyz_data[0][:, i, :, :][0]

    x_0 = int(float(frap_position['CenterX']))
    y_0 = int(float(frap_position['CenterY']))
    radius = int(float(frap_position['Radius']))
    distance_from_border = 1
    # make radius slighly smaller so border is not in range
    measurement_radius = radius - distance_from_border  # [index]
    pixels_in_circle = []

    for x in range(x_0 - measurement_radius, x_0 + measurement_radius):
        for y in range(y_0 - measurement_radius, y_0 + measurement_radius):
            dx = x - int(float(frap_position['CenterX']))
            dy = y - int(float(frap_position['CenterY']))
            distanceSquared = dx ** 2 + dy ** 2
            # print (distanceSquared)
            if distanceSquared <= (measurement_radius ** 2):
                pixel_val = measurement_image[y][x]
                pixels_in_circle.append(pixel_val)
    I_mean[i - bleach_frame] = np.mean(pixels_in_circle)

#ic(I_mean)

# Compute normalized intensity
I_norm = I_mean / I_0
I_pre_norm = I_pre / I_0
#ic(I_norm)
#ic(I_pre_norm)

# Plot normalized intensity
plt.plot(t_pre, I_pre_norm, 'k-')
plt.plot(t, I_norm, 'k-')
plt.xlabel('time (s)')
plt.ylabel(r'$I_\mathrm{norm}$')
plt.show()






# Define log posterior
def log_posterior(p, I_norm, t, d_x, d_y):
    """
    Return log of the posterior.
    """
    return -len(I_norm) / 2.0 * np.log(
            ((I_norm - norm_fluor_recov(p, t, d_x, d_y))**2).sum())

# Define fit function
def norm_fluor_recov(p, t, d_x, d_y):
    """
    Return normalized fluorescence as function of time.
    """
    # Unpack parameters
    f_b, f_f, D, k_off = p

    # Function to compute psi
    def psi(t, D, d_i):
        return d_i / 2.0 * scipy.special.erf(d_i / np.sqrt(4.0 * D * t)) \
                     - np.sqrt(D * t / np.pi) \
                                * (1.0- np.exp(-d_i**2 / (4.0 * D * t)))
    psi_x = psi(t, D, d_x)
    psi_y = psi(t, D, d_y)
    return f_f * (1.0 - f_b * 4.0 * np.exp(-k_off * t) / d_x / d_y
                            * psi_x * psi_y)

# Define residual
def resid(p, I_norm, t, d_x, dy):
    return I_norm - norm_fluor_recov(p,t, d_x, d_y)


d_x=160
d_y=180

# Perform the curve fit
p0 = np.array([0.9, 0.9, 10.0, 0.1])
popt, junk_output = scipy.optimize.leastsq(resid, p0,
                                           args=(I_norm, t, d_x, d_y))

# # Compute the covariance
# cov = np.cov()
# hes = jb.hess_nd(log_posterior, popt, args=(I_norm, t, d_x, d_y))
# cov = -np.linalg.inv(hes)

# Report results
formats = tuple(popt)# + tuple(np.sqrt(np.diag(cov)))
print("""
f_f = {0:.3f} +- {4:.3f}
f_b = {1:.3f} +- {5:.3f}
D = {2:.3f} +- {6:.3f} µm^2/s
k_off = {3:.3f} +- {7:.3f} (1/s)
""".format(*formats))

# Plot recovery trace
plt.plot(t, I_norm, 'k-')
plt.plot(t, norm_fluor_recov(popt, t, d_x, d_y), 'r-')
plt.xlabel('time (s)')
plt.ylabel('norm. fluor. (a.u.)')