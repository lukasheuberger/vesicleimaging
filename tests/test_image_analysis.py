"""Plotting of czi images and image analysis."""
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from icecream import ic
import pandas as pd
from matplotlib_scalebar.scalebar import ScaleBar
import czi_image_handling as handler
from numba import njit
import cProfile
from skimage import filters, feature, draw
from scipy.ndimage import gaussian_filter


def test_all_functions(path):
    """
    The test_all_functions function is a test function that runs
    all other functions in this file. It takes one argument,
    which is the path to where test images are located.


    Args:
        path: Specify the path to the directory containing test files
    """

    from czi_image_handling import load_h5_data, disp_basic_img_info
    os.chdir(path)
    path = os.getcwd()
    ic(path)

    image_data, metadata, add_metadata = load_h5_data(path)
    disp_basic_img_info(image_data, metadata)
    test_index = 0

    plot_images(image_data[test_index], metadata[test_index],
                add_metadata[test_index], saving=True)

    param1_array = [10]
    param2_array = [150]
    minmax = [30, 60]
    display_channel = 0
    detection_channel = 0

    test_data = [image_data[test_index]]
    test_metadata = [metadata[test_index]]

    detected_circles = detect_circles(test_data, test_metadata,
                                      param1_array=param1_array,
                                      param2_array=param2_array,
                                      minmax=minmax,
                                      display_channel=display_channel,
                                      detection_channel=detection_channel)

    measurement_channel = 0
    df, circles = measure_circle_intensity(test_data, test_metadata, detected_circles,
                                            measurement_channel, excel_saving=False)

if __name__ == '__main__':
    # path = input('path to data folder: ')
    DATA_PATH = '../test_data/general'

    # profiler = cProfile.Profile()
    # profiler.enable()


    test_all_functions(DATA_PATH)

    # profiler.disable()
    # profiler.print_stats(sort='cumulative')
# todo make all this in one file that just contains all functions from the other files