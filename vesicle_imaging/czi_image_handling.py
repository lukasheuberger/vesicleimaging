"""Basic czi image handling and information extraction."""
import os

import xml.etree.ElementTree as ET
import czifile
from icecream import ic
import cv2
from skimage import img_as_ubyte
import imgfileutils as imf


def get_files(path: str):
    """
    The get_files function returns a list of all the .czi files in the
    directory and sorts them by name. It also returns a list of
    just the filenames, sorted in reverse order so that they match
    up with their corresponding filepaths.

    Args:
        path:str: Specify the path to the directory containing the

    Returns:
        A list of the files in the folder and a list of the filenames
    """
    files_array = []
    filenames = []
    # r=root, d=directories, f = files
    for root, dirs, files in os.walk(path):
        files.sort()
        for file in files:
            if '.czi' in file:
                if not file.startswith('.'):
                    filenames.append(file)
                    file_path = os.path.join(root, file)
                    files_array.append(file_path)

    filenames.sort(reverse=True)
    files_array.sort(reverse=True)

    return files_array, filenames


def write_metadata_xml(path: str, files: list):
    """
    The write_metadata_xml function creates a folder named
    'metadata' in the path of the czi file. It then creates
    an XML file with all metadata from the czi file and
    saves it to this new folder.

    Args:
        path:str: Define the path to the folder
                    containing all of your czi files
        files:list: Pass the list of files to be processed

    Returns:
        The path to the xml file and the filename of that xml
    """
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


def load_image_data(files: list, write_metadata: bool = False):
    """
    The load_image_data function loads the image data from a list of files.
    It returns three lists: all_img_data, all_metadata,
    and all_additional metadata. The first two are lists of numpy arrays
    containing the image data and metadata for each file in files.
    The third is a list of dictionaries containing additional
    information about each file.

    Args:
        files:list: Pass a list of files to the function
        write_metadata:bool=False: Write the metadata to a file

    Returns:
        Three values: all_img_data, all_metadata, and all_additional metadata
    """
    all_img_data = []
    all_metadata = []
    all_add_metadata = []

    for file in files:
        # get the array and the metadata
        print(file)
        img_data, metadata, add_metadata = imf.get_array_czi(
            file, return_addmd=False)
        all_img_data.append(img_data)
        all_metadata.append(metadata)
        all_add_metadata.append(add_metadata)

    if write_metadata:
        path = os.path.dirname(files[0])
        ic(path)
        write_metadata_xml(path, files)

    return all_img_data, all_metadata, all_add_metadata


def extract_channels(img_data: list[int], img_type: str,
                     channels: list[int] = None):
    """
    The extract_channels function extracts the channels from a list of images.
    The function takes in a list of images and an image type
    (zstack, timelapse, or image). If the image type is zstack,
    it extracts only one channel from each z-slice. If the
    image type is timelapse or image, it extracts only one
    channel from each timepoint/image.

    Args:
        img_data:list[int]: list of images to the extract_channels function
        img_type:str:  dimension of the image used for stacking
        channels=None: channels to extract from the timelapse images

    Returns:
        A list of images with the specified channels
    """

    # shape: B, V, C, T, Z, Y, X'

    extracted_channels = []
    for image in img_data:
        ic(image.shape)
        if img_type == 'zstack':
            extracted_channels.append(image[0, 0, :, 0, :, :, :])
        elif img_type == 'zstack':
            extracted_channels.append(image[0, 0, :, 0, :, :, :])
        elif img_type == 'image':
            extracted_channels.append(image[0, 0, 0, 0, :, :, :])
        elif img_type == 'timelapse':
            # check if timelapse needs to be reduced to certain channels
            if channels is not None:
                extracted_channels.append(image[0, 0, channels, :, 0, :, :])
            else:
                extracted_channels.append(image[0, 0, :, :, 0, :, :])
        else:
            raise ValueError(f'unknown image type: {img_type}')

    return extracted_channels


def disp_channels(add_metadata):
    """
    The disp_channels function displays the channels of each image in a list.
    The function takes as input add_metadata, which is a list of dictionaries
    containing metadata for all images. Each dictionary contains keys that
    are the names of different fields and values that are lists with
    information about those fields.
    The first element in add_metadata is the dictionary for image 0,
    so we use index 0 to access it and its Experiment field (which
    contains more nested dictionaries).
    We then access its ExperimentBlocks field (also a dict) and from
    there we get its AcquisitionBlock field (another dict),
    whose MultiTrackSetup has TrackSetups as values

    Args:
        add_metadata: Get the metadata from the file

    Returns:
        A list of the channels used in the experiment
    """

    # channels are the same for both conditions
    add_metadata_detectors = add_metadata[0]['Experiment']\
        ['ExperimentBlocks']['AcquisitionBlock']['MultiTrackSetup']\
        ['TrackSetup'][0]['Detectors']['Detector']

    channel_names = []
    dyes = []
    detectors = []

    print('Channels')
    print('------------------------------------')

    # channels of all images are the same so image 0 taken
    for index, channel in enumerate(add_metadata_detectors):
        print(f'IMAGE {index}:')
        detectors.append(channel['ImageChannelName'])
        dyes.append(channel['Dye'])
        channel_name = ' '.join([channel['ImageChannelName'],
                                 str(channel['Dye'])])
        channel_names.append(channel_name)

        print(channel['ImageChannelName'])
        print(channel['Dye'])
        print(channel_name)
        print('------------------------------------')

    return detectors, channel_names, dyes


def disp_all_metadata(metadata):
    """
    The disp_all_metadata function displays all the metadata
    for each image in a list of dictionaries.
    It takes one argument, metadata, which is a list of dictionaries.

    Args:
        metadata: Store the metadata of all images in a list

    Returns:
        All the metadata for each image in a list
    """

    # show all the metadata
    for index, image in enumerate(metadata):
        print(f'Image {index}:')
        for key, value in image.items():
            # print all key-value pairs for the dictionary
            print(key, ' : ', value)
        print('------------------------------------')


def disp_basic_img_info(img_data, img_metadata):
    """
    The disp_basic_img_info function displays the
    following information about each image:
        - Filename
        - Shape of the CZI array (B, V, C, T, Z, Y and X)

    Args:
        img_data: Store the image data
        img_metadata: Store the metadata of each image

    Returns:
        The basic information about the image data and metadata
    """

    for index in range(len(img_data)):
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


def disp_scaling(img_add_metadata):
    """
    The disp_scaling function returns a list of the
    scaling values for each image.
    The scaling value is the pixel size in microns per pixel.


    Args:
        img_add_metadata: Pass the metadata of the image to

    Returns:
        A list of the scaling_x values from each image
    """

    scaling_x = []
    for index, image in enumerate(img_add_metadata):
        scale = image['Experiment']['ExperimentBlocks']\
            ['AcquisitionBlock']['AcquisitionModeSetup']['ScalingX']
        scaling_x.append(scale)

    return scaling_x


def increase_brightness(img, value=30):
    """
    The increase_brightness function takes an image and increases
    the brightness of the image by a specified value. The function
    first converts the input image to HSV color space,
    then splits it into its hue, saturation, and value channels.
    The function then iterates through each pixel in the V
    channel (the Value or brightness channel). If any pixels have a
    value greater than 255-value (where 255 is maximum brightness),
    they are set to 255. Otherwise, they are increased by the value
    parameter. The channels are merged together and converted
    back to BGR before returning.

    Args:
        img: Pass the image to be adjusted
        value=30: Increase the brightness of the image

    Returns:
        The image with the brightness increased by the value specified
    """

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def convert8bit(img: list[int]):
    """
    The convert8bit function converts the image from uint16 to uint8.

    Args:
        img:list[int]: Pass the image data to the function

    Returns:
        A uint8 type array
    """
    print(img.dtype)
    print('image is uint16, converting to uint8 ...')
    # img8bit = cv2.convertScaleAbs(img, alpha=(255.0/65535.0))
    img8bit = img_as_ubyte(img)
    print('done converting to uint8')

    return img8bit


def test_all_functions(path):
    """
    The test_all_functions function is a test
    function that runs all of the functions in this module.
    It takes one argument, which is the path to the
    directory containing all of your images.


    Args:
        path: Specify the path to the folder containing

    Returns:
        A tuple of the following:
    """

    files, filenames = get_files(path)
    ic(filenames)
    write_metadata_xml(path, files)
    img_data, metadata, add_metadata = load_image_data(files,
                                                       write_metadata=False)

    # ic(img_data[0].shape)
    disp_basic_img_info(img_data, metadata)
    img_reduced = extract_channels(img_data, img_type='timelapse')
    ic(img_reduced[0].shape)

    # disp_all_metadata(metadata)

    disp_channels(add_metadata)

    disp_scaling(add_metadata)


if __name__ == '__main__':
    # path = input('path to data folder: ')
    # DATA_PATH = '/Users/heuberger/code/vesicle-imaging/test_data/general'
    DATA_PATH = '/Users/lukasheuberger/code/phd/vesicle-imaging/test_data/general'
    test_all_functions(DATA_PATH)

    # todo: function that writes image data to hdf5
