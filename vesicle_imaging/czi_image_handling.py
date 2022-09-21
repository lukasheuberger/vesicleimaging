"""Basic czi image handling and information extraction."""

import czifile
import os
import xml.etree.ElementTree as ET
import imgfileutils as imf
from icecream import ic
import cv2
import numpy as np
from skimage import img_as_ubyte

def get_files(path: str):
    files = []
    filenames = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        f.sort()
        for file in f:
            if '.czi' in file:
                if not file.startswith('.'):
                    filenames.append(file)
                    file_path = os.path.join(r, file)
                    files.append(file_path)

    filenames.sort(reverse=True)
    files.sort(reverse=True)

    return files, filenames


def write_metadata_xml(path: str, files: list):
    try:
        metadata_path = ''.join([path, '/metadata'])
        ic(metadata_path)
        os.mkdir(metadata_path)
    except FileExistsError:
        pass

    for file in files:
        ic(file)
        xmlczi = czifile.CziFile(file).metadata()

        # define the new filename for the XML to be created later
        # split string at last / and add folder
        xmlfile = file.replace('.czi', '_CZI_MetaData.xml')
        xmlfile, filename = xmlfile.rsplit('/', 1)
        xmlfile = ''.join([xmlfile, '/metadata/', filename])

        # get the element tree
        tree = ET.ElementTree(ET.fromstring(xmlczi))

        # write xml to disk
        tree.write(xmlfile, encoding='utf-8', method='xml')

        print('Write special CZI XML metainformation for: ', xmlfile)


def load_image_data(files: list):
    all_img_data = []
    all_metadata = []
    all_add_metadata = []

    for file in files:
    # get the array and the metadata
        print (file)
        img_data, metadata, add_metadata = imf.get_array_czi(file, return_addmd=False)
        all_img_data.append(img_data)
        all_metadata.append(metadata)
        all_add_metadata.append(add_metadata)
    return all_img_data, all_metadata, all_add_metadata

def extract_channels(img_data: list[int], type: str, channels = None):
    # shape: B, V, C, T, Z, Y, X'

    extracted_channels = []
    for image in img_data:
        ic(image.shape)
        if type == 'zstack':
            extracted_channels.append(image[0, 0, :, 0, :, :, :])

        elif type == 'zstack':
            extracted_channels.append(image[0, 0, :, 0, :, :, :])
        elif type == 'image':
             extracted_channels.append(image[0, 0, 0, 0, :, :, :])
        elif type == 'timelapse':
            # check if timelapse needs to be reduced to certain channels
            if channels is not None:
                extracted_channels.append(image[0, 0, channels, :, 0, :, :])
            else:
                extracted_channels.append(image[0, 0, :, :, 0, :, :])
        else:
            raise ValueError('unknown image type: %s' % type)

    return extracted_channels

def disp_channels(add_metadata):#### fix this and combine with bottom one!
    # channels are the same for both conditions
    channel_names = []
    dyes = []
    add_metadata_detectors = add_metadata['Experiment']['ExperimentBlocks']['AcquisitionBlock']['MultiTrackSetup']['TrackSetup']['Detectors']['Detector']
    # channels of all images are the same so image 0 taken
    for channel in add_metadata_detectors:
        print(channel['ImageChannelName'])
        print(channel['Dye'])
        dyes.append(channel['Dye'])
        channel_name = ' '.join([channel['ImageChannelName'], str(channel['Dye'])])
        print(channel_name)
        channel_names.append(channel_name)
        print('------------------------------------')
    return dyes


def disp_all_metadata(metadata):
    # show all the metadata
    for index, image in enumerate(metadata):
        for key, value in image.items():
            # print all key-value pairs for the dictionary
            print(key, ' : ', value)
        print('------------------------------------')


def disp_basic_img_info(img_data, img_metadata):
    for index, img in enumerate(img_data):
        print(f'Image {index + 1}:')
        print('Filename: ', img_metadata[index]['Filename'])
        print('CZI Array Shape : ', img_metadata[index]['Shape_czifile'])
        print('CZI Dimension Entry : ', img_metadata[index]['DimOrder CZI'])
        print('-----------------------------')
    print('shape: B, V, C, T, Z, Y, X, 0')
    print("""
    B - aquisition block index in segmented experiments
    V - View index (for multi – view images, e.g. SPIM)
    C - Channel in a Multi-Channel data set
    T - Time point in a sequentially acquired series of data.
    Z - Slice index (Z – direction).
    Y - Pixel index / offset in the Y direction
    X - Pixel index / offset in the X direction
    """)

def xxxdisp_channels(img_metadata):
    #channels = img_metadata[0][0]['ChannelNames']
    #channels = img_metadata[0]['ChannelNames']
    ic(img_metadata[0][0]['ChannelNames'])
    channels = img_metadata[0][0][0]['ChannelNames']
    #dyes = img_metadata[0][0]['Channels']
    dyes = img_metadata[0]['Channels']
    return channels, dyes


def disp_scaling(img_add_metadata):
    scaling_x = []
    for index, image in enumerate(img_add_metadata):
        scale = image[0]['Experiment']['ExperimentBlocks']['AcquisitionBlock']['AcquisitionModeSetup']['ScalingX']
        #scale = image['Experiment']['ExperimentBlocks']['AcquisitionBlock']['AcquisitionModeSetup']['ScalingX']
        #todo fix somehow that this doesn't have to be changed all the time
        scaling_x.append(scale)
    # print('scale factor: ', scaling_x)
    return scaling_x


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def convert8bit(img: list[int]):
    print(img.dtype)
    print('image is uint16, converting to uint8 ...')
    # img8bit = cv2.convertScaleAbs(img, alpha=(255.0/65535.0))
    img8bit = img_as_ubyte(img)
    print('done converting to uint8')
    return img8bit


def test_all_functions(path):
    files, filenames = get_files(path)
    write_metadata_xml(path, files)
    img_data, metadata, add_metadata = load_image_data(files)

    # ic(img_data[0].shape)
    disp_basic_img_info(img_data, metadata)
    img_reduced = extract_channels(img_data, type = 'timelapse')
    ic(img_reduced[0].shape)

    disp_all_metadata(metadata)

    disp_channels(add_metadata) #todo needs to be fixed

if __name__ == '__main__':
    # path = input('path to data folder: ')
    DATA_PATH = '/Users/heuberger/code/vesicle-imaging/test_data/general'
    test_all_functions(DATA_PATH)

    # todo: function that writes image data to hdf5
