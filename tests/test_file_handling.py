"""Basic czi image handling and information extraction."""
import os
import pickle
import xml.etree.ElementTree as ET

import cv2
import czifile
import h5py
import numpy as np
from icecream import ic
from skimage import img_as_ubyte
import cProfile

import imgfileutils as imf
import concurrent.futures








def test_all_functions(path):
    """
    The test_all_functions function is a test
    function that runs all of the functions in this module.
    It takes one argument, which is the path to the
    directory containing all of your images.


    Args:
        path: Specify the path to the folder containing
    """
    os.chdir(path)
    path = os.getcwd()
    ic(path)

    files, filenames = get_files(path)
    ic(filenames)
    write_metadata_xml(path, files)
    img_data, metadata, add_metadata =\
        load_image_data(files, write_metadata=False)

    # ic(img_data[0].shape)
    disp_basic_img_info(img_data, metadata)
    img_reduced = extract_channels(img_data)

    disp_all_metadata(metadata)

    ic(get_channels(add_metadata))

    ic(disp_scaling(add_metadata))

    save_files(img_reduced, metadata, add_metadata)

    load_h5_data(path)


if __name__ == '__main__':
    # path = input('path to data folder: ')
    DATA_PATH = '../test_data/general'

    # profiler = cProfile.Profile()
    # profiler.enable()

    test_all_functions(DATA_PATH)

    # profiler.disable()
    # profiler.print_stats(sort='cumulative')

