import sys

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
frap_position = image_add_metadata[0][0]['Layers']['Layer'][0]['Elements']['Circle']['Geometry'] #centerx, centery, radius
deltaT = image_add_metadata[0][0]['DisplaySetting']['Information']['Image']['Dimensions']['Channels']['Channel'][0]['LaserScanInfo']['FrameTime']
ic(deltaT)
scaling = float(handler.disp_scaling(image_add_metadata)[0])
ic(scaling)

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
radius_px = int(float(frap_position['Radius'])) #todo correct rounding
ic(x_0, y_0, radius_px)
radius_m = float(frap_position['Radius'])*scaling # in m
radius_um = radius_m * 10**6 # convert to um
ic(radius_px, radius_m, radius_um)

for frame in range(0, num_frames):
    #ic(frame)
    measurement_image = image_cxyz_data[0][:,frame,:,:][0].copy()

    #blank_image = np.zeros((256,256,3), np.uint8)

    # make radius slighly smaller so border is not in range
    distance_from_border = 0
    measurement_radius = radius_px - distance_from_border  # [index]
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
    # print('min: ', np.min(pixels_in_circle))
    # print('max: ', np.max(pixels_in_circle))
    # print('average: ', np.mean(pixels_in_circle))
    # print('stdev: ', np.std(pixels_in_circle))
    stats.append(np.mean(pixels_in_circle))

#ic(stats)
plt.figure()
plt.plot(stats)
plt.title('stats')
plt.show()


bleach1 = stats[0:5]
recovery1 = stats[5:100]
t1=range(5,100)
#t1=list(range(5,100))
t1 = [timestep * float(deltaT) for timestep in t1] # convert to actual seconds
bleach2 = stats[100:105]
recovery2 = stats[105:200]
recovery1_norm = recovery1/np.mean(bleach1)
recovery2_norm = recovery2/np.mean(bleach2)

plt.figure()
plt.title('stats not normalized')
plt.plot(bleach1, label = 'run 1')
plt.plot(recovery1, label = 'run 1')
plt.plot(bleach2, label = 'run 2')
plt.plot(recovery2, label = 'run 2')
plt.legend()
plt.show()

plt.figure()
plt.title('stats normalized')
plt.plot(recovery1_norm, label = 'run 1')
plt.plot(recovery2_norm, label = 'run 2')
plt.legend()
plt.show()

#calculate area of circle
ic(float(frap_position['Radius']))
area = radius_um**2*np.pi #in um^2
d_x = np.sqrt(area)
d_y = np.sqrt(area)
ic(area, d_x, d_y)

# Define log posterior
def log_posterior(p, I_norm, t, d_x, d_y):
    """
    Return log of the posterior.
    """
    return -len(I_norm) / 2.0 * np.log(
            ((I_norm - norm_fluor_recov(p, t, d_x, d_y))**2).sum())

# Define fit function
def norm_fluor_recov(p, time, d_x, d_y):
    """
    Return normalized fluorescence as function of time.
    """
    # Unpack parameters
    #f_b, f_f, D, k_off = p
    f_f, f_b, D, k_off = p

    # Function to compute psi
    def psi(time_array, D, d_i):
        return [d_i / 2.0 * scipy.special.erf(d_i / np.sqrt(4.0 * D * t)) \
                     - np.sqrt(D * t / np.pi) \
                                * (1.0- np.exp(-d_i**2 / (4.0 * D * t))) for t in time_array]
    psi_x = psi(time, D, d_x)
    psi_y = psi(time, D, d_y)
    return f_f * (1.0 - f_b * 4.0 * np.exp([-k_off * t for t in time]) / d_x / d_y * psi_x * psi_y)
    #todo: make consistent with naming t vs t_array

# Define residual
def resid(p, I_norm, t, d_x, d_y):
    return I_norm - norm_fluor_recov(p,t, d_x, d_y)

# Perform the curve fit
p0 = np.array([0.9, 0.9, 1.0, 0.1])
#p0 = np.array([0.9, 0.9, 10.0, 0.1])
popt, pcov = scipy.optimize.leastsq(resid, p0, args=(recovery1_norm, t1, d_x, d_y))
#popt, junk_output = scipy.optimize.leastsq(resid, p0, args=(I_norm, t, d_x, d_y))
ic(popt)
s_sq = (resid(popt, recovery1_norm, t1, d_x, d_y)**2).sum()/(len(recovery1_norm)-len(p0))
pcov = pcov * s_sq
ic(s_sq, pcov)


# errfunc = lambda p, x, y: norm_fluor_recov(x,p) - y
# s_sq = (errfunc(popt, t1, recovery1_norm)**2).sum()/(len(recovery1_norm)-len(p0))
# pcov = pcov * s_sq
# perr = np.sqrt(np.diag(pcov))
# ic(pcov, perr)

# Compute the covariance
#hes = jb.hess_nd(log_posterior, popt, args=(I_norm, t, d_x, d_y))
#cov = -np.linalg.inv(hes)

# Report results
formats = tuple(popt) #+ tuple(np.sqrt(np.diag(pcov)))
# print("""
# f_f = {0:.3f} +- {4:.3f}
# f_b = {1:.3f} +- {5:.3f}
# D = {2:.3f} +- {6:.3f} µm^2/s
# k_off = {3:.3f} +- {7:.3f} (1/s)
# """.format(*formats))
print("""
f_f = {0:.3f}
f_b = {1:.3f}
D = {2:.3f} µm^2/s
k_off = {3:.3f} (1/s)
""".format(*formats))
#todo: make that this works for covariance
#todo: also, D might be wrong

# Plot recovery trace
plt.plot(t1, recovery1_norm, 'k-')
plt.plot(t1, norm_fluor_recov(popt, t1, d_x, d_y), 'r-')
plt.xlabel('time (s)')
plt.ylabel('norm. fluor. (a.u.)')
plt.show()


def fit(d0, t):#, radius):
    tau = d0
    #ic(D)
    #ic(tau)
    #F(t) = np.exp(-2*tau/t) * (scipy.special.i0(2*tau/t) + scipy.special.i1(2*tau/t))
    #return np.exp(-2*tau/t) * (scipy.special.i0(2*tau/t) + scipy.special.i1(2*tau/t))
    #return np.exp(-2*((radius ** 2) / (4 * D))/t) * (scipy.special.i0(2*(radius ** 2 / (4 * D))/t) + scipy.special.i1(2*(radius ** 2 / (4 * D))/t))
    return np.exp(-2*tau/t) * (scipy.special.i0(2*tau/t) + scipy.special.i1(2*tau/t)) #fast bessel
    #return np.exp(-2*tau/t) * (scipy.special.iv(0, 2*tau/t) + scipy.special.iv(1, 2*tau/t)) #traditional bessel

def newresid(d0, I_norm, t):
    return I_norm - fit(d0, t)

d0 = [10]#[0.9, 0.9, 10.0, 0.1])
coeffs, _ = scipy.optimize.leastsq(newresid, d0, args=(recovery1_norm, t1))
ic(coeffs)
diffusion_const = radius_um ** 2 / (4 * coeffs)
ic(diffusion_const)

plt.plot(t1, recovery1_norm, 'k-')
plt.plot(t1, fit(coeffs, t1), 'r-')
plt.xlabel('time (s)')
plt.ylabel('norm. fluor. (a.u.)')
plt.show()