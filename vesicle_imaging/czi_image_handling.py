"""Basic czi image handling and information extration."""


import czifile
import os
import xml.etree.ElementTree as ET
import imgfileutils as imf
from icecream import ic
import cv2
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
    return files, filenames


def write_metadata_xml(path: str, files: list):
    try:
        metadata_path = ''.join([path, 'metadata'])
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

        # xmlfile = ''.join(['metadata/',xmlfile])

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


def extract_channels_xyz(img_data: list):
    img_xy_data = []
    for index, img in enumerate(img_data):
        channels_xy = []
        for image in img:
            channels_xy.append(image[0, 0, :, 0, :, :, :])
            #channels_xy.append(image[:, :, :])
        img_xy_data.append(channels_xy)
    print ('image XY data extracted')
    return img_xy_data


def extract_channels_timelapse(img_data):
    channels_timelapse = []
    for image in img_data:
        channels_timelapse.append(image[0, 0, :, :, 0, :, :])
    return channels_timelapse


def disp_channels(add_metadata):
    # channels are the same for both conditions
    channel_names = []
    dyes = []
    add_metadata_detectors = \
    add_metadata[0]['Experiment']['ExperimentBlocks']['AcquisitionBlock']['MultiTrackSetup']['TrackSetup'][
        'Detectors']['Detector']
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
        for key, value in image[0].items():
            # print all key-value pairs for the dictionary
            print(key, ' : ', value)
        print('------------------------------------')


def disp_basic_img_info(img_data, img_metadata):
    for index, img in enumerate(img_data):
        image = img[0]
        print('image', index + 1, ':')
        print('Image: ', img_metadata[index][0]['Filename'])
        print('CZI Array Shape : ', img_metadata[index][0]['Shape_czifile'])
        print('CZI Dimension Entry : ', img_metadata[index][0]['DimOrder CZI'])
        print('-----------------------------')


def disp_channels(img_metadata):
    channels = img_metadata[0][0]['ChannelNames']
    dyes = img_metadata[0][0]['Channels']
    return channels, dyes


def disp_scaling(img_add_metadata):
    scaling_x = []
    for index, image in enumerate(img_add_metadata):
        # scale = image[0]['Experiment']['ExperimentBlocks']['AcquisitionBlock']['AcquisitionModeSetup']['ScalingX']
        scale = image[0]['Experiment']['ExperimentBlocks']['AcquisitionBlock']['AcquisitionModeSetup']['ScalingX']
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


def convert8bit(img):
    print(img.dtype)
    print('image is uint16, converting to uint8 ...')
    #img8bit = cv2.convertScaleAbs(img, alpha=(255.0/65535.0))
    img8bit = img_as_ubyte(img)
    print('done converting to uint8')
    return img8bit