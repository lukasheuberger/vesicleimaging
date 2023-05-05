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

    # Load the data
    os.chdir(path)
    # ic(os.listdir(path))
    # ic(os.getcwd())

    files_array = []
    filenames = []
    # r=root, d=directories, f = files
    for root, _, files in os.walk(path):
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


def write_metadata_xml(path: str,
                       files: list):
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

    # TODO: doesn't work if files and path are not the same
    #  (-> files in subfolders)

    metadata_path = ''.join([path, '/metadata'])
    ic(metadata_path)

    try:
        os.mkdir(metadata_path)
        print(f'new folder created: {metadata_path}')
    except FileExistsError:
        print(f'folder already exists: {metadata_path}')

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
    # todo make that this only takes czi images

    all_img_data = []
    all_metadata = []
    all_add_metadata = []

    files = [file for file in files if file.endswith('.czi')]

    def process_file(file):
        print(file)
        img_data, metadata, add_metadata = imf.get_array_czi(file, return_addmd=False)
        return img_data, metadata, add_metadata

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_file, files))

    for img_data, metadata, add_metadata in results:
        all_img_data.append(img_data)
        all_metadata.append(metadata)
        all_add_metadata.append(add_metadata)

    if write_metadata:
        path = os.path.dirname(files[0])
        ic(path)
        write_metadata_xml(path, files)

    return all_img_data, all_metadata, all_add_metadata


def load_image_data_old(files: list,
                    write_metadata: bool = False):
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

    ic(files)

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


def extract_channels(img_data: list[int]):
    """
    The extract_channels function extracts the first channel
     of each image in a list of images.

    Args:
        img_data:list[int]: Store the image data

    Returns:
        A list of images that have been extracted from the original image
    """

    # shape: B, V, C, T, Z, Y, X'
    # TODO FIX ME
    extracted_channels = []
    for image in img_data:
        try:
            extracted_channels.append(image[0, 0, :, :, :, :, :])
        except (AttributeError, TypeError):
            extracted_channels.append(image[0][0, 0, :, :, :, :, :])

    return extracted_channels


def get_channels(add_metadata):
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
    # TODO fix this for airy scan
    # channels are the same for both conditions

    try:
        add_metadata_detectors = add_metadata[0]['Experiment']\
            ['ExperimentBlocks']['AcquisitionBlock']['MultiTrackSetup']\
            ['TrackSetup'][0]['Detectors']['Detector']
    except KeyError:
        # maybe this zero also needs to be removed to catch all cases
        add_metadata_detectors = add_metadata[0]['Experiment'] \
            ['ExperimentBlocks']['AcquisitionBlock']['MultiTrackSetup'] \
            ['TrackSetup']['Detectors']['Detector']
    except TypeError:
        add_metadata_detectors = add_metadata[0][0]['Experiment'] \
            ['ExperimentBlocks']['AcquisitionBlock']['MultiTrackSetup'] \
            ['TrackSetup']['Detectors']['Detector']

    # ic(add_metadata_detectors)

    channel_names = []
    dyes = []
    detectors = []

    print('Channels')
    print('------------------------------------')

    if isinstance(add_metadata_detectors, list) is False:
        # airyscan has non-list metadata here, so convert to list
        add_metadata_detectors = [add_metadata_detectors]

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


def disp_basic_img_info(img_data: list,
                        img_metadata: list):
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

    print('------------------------------------')
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
    for image in img_add_metadata:
        try:
            scale = image['Experiment']['ExperimentBlocks'] \
                ['AcquisitionBlock']['AcquisitionModeSetup']['ScalingX']
        except TypeError:
            scale = image[0]['Experiment']['ExperimentBlocks'] \
                ['AcquisitionBlock']['AcquisitionModeSetup']['ScalingX']

        scaling_x.append(scale)

    return scaling_x


def increase_brightness(img,
                        value):
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
    hue, sat, val = cv2.split(hsv)

    lim = 255 - value
    val[val > lim] = 255
    val[val <= lim] += value

    final_hsv = cv2.merge((hue, sat, val))
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
    print(f'image is {img.dtype}, converting to uint8 ...')
    # img8bit = cv2.convertScaleAbs(img, alpha=(255.0/65535.0))
    img8bit = img_as_ubyte(img)
    print('done converting to uint8')

    return img8bit


def save_files(data: list[int],
               metadata: list[str],
               add_metadata: list[str]):
    """
    The save_files function takes in a list of images, and saves them to
    hdf5 files. The function also takes in a list of metadata and
    additional metadata and saves them to a pickle file.

    Args:
        data:list[int]: Store the image data in a list
        metadata:list[str]: Store the metadata of each image in a list
        add_metadata:list[str]: Save additional metadata to the hdf5 file

    Returns:
        A list of filenames that have been saved to hdf5
    """

    # create a temporary list of filename for storing in hdf5
    filenames = []
    for image in metadata:
        filenames.append(image['Filename'])

    for index, image in enumerate(data):
        temp_filename_base = filenames[index].split('.')[0]
        temp_filename_h5 = temp_filename_base + '.h5'
        # ic(temp_filename_base)

        h5file = h5py.File(temp_filename_h5, 'w')
        h5file.create_dataset('filename', data=filenames[index])
        h5file.create_dataset('image', data=image)
        h5file.close()
        print(f'hdf5 file {temp_filename_h5} created successfully.')

        # save metadata to pickle file
        temp_filename_metadata = temp_filename_base + '_metadata.pkl'
        with open(temp_filename_metadata, 'wb') as metadata_pickle:
            pickle.dump(metadata[index], metadata_pickle)
        print(f'    - pickle file '
              f'{temp_filename_metadata} created successfully.')

        # save additional metadata to pickle file
        temp_filename_add_metadata = temp_filename_base + '_addmetadata.pkl'
        with open(temp_filename_add_metadata, 'wb') as add_metadata_pickle:
            pickle.dump(add_metadata[index], add_metadata_pickle)
        print(f'    - pickle file '
              f'{temp_filename_add_metadata} created successfully.')


def load_h5_data(path: str):
    """
    The load_h5_data function loads the data from a given path.
    It returns three lists: image_data, metadata and add_metadata.
    The image_data list contains all the images in numpy array format,
    the metadata list contains all the metadata in dictionary format
    and add_metadata is a list of dictionaries containing additional
    information about each image.

    Args:
        path:str: Specify the path to the folder containing data files

    Returns:
        A tuple of three lists
    """

    image_data = []
    metadata = []
    add_metadata = []
    filenames = []

    # ic(path)
    # ic(os.listdir(path))

    for file in os.listdir(path):
        if file.endswith(".h5"):
            filename = file.split('.')[0]
            print(f'loading {file} ...')

            h5file = h5py.File(file, 'r')
            # ic(h5file.keys())
            data = np.array(h5file.get('image'))
            image_data.append(data)
            h5file.close()

            filenames.append(filename)

    for filename in filenames:
        # if file.endswith("_metadata.pkl"):
        filename_metadata = ''.join([filename, '_metadata.pkl'])
        with open(filename_metadata, "rb") as metadata_pickle:
            meta = pickle.load(metadata_pickle)
            metadata.append(meta)

        filename_add_metadata = ''.join([filename, '_addmetadata.pkl'])
        # if file.endswith("_addmetadata.pkl"):
        with open(filename_add_metadata, "rb") as add_metadata_pickle:
            add_meta = pickle.load(add_metadata_pickle)
            add_metadata.append(add_meta)

    return image_data, metadata, add_metadata

def max_projection(image_data: list[int]):
    """
    The max_projection function takes in a list of images and
    returns a single image that is the maximum projection of
    all the images in the list.
    Args:
        image_data:list[int]: Pass a list of images
    Returns:
        A single image that is the maximum projection of all the images
        in the list
    """
    # todo improve this
    # todo add this to tests
    new_image = []
    ic(image_data.shape)
    for channel_index, channel_img in enumerate(image_data):
        ic(channel_index, channel_img.shape)
        for timepoint_index, timepoint in enumerate(channel_img):
            ic(timepoint_index, timepoint.shape)
            max_projection = np.max(timepoint, axis=0)
            ic(max_projection.shape)
        new_image.append(max_projection)
    return new_image


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

