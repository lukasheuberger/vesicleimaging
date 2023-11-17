# Description: This file contains functions for processing czi files.

import os
from .imgfileutils import get_array_czi
import concurrent.futures
import czifile
import xml.etree.ElementTree as ET
import multiprocessing

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
    print(f'path to metadata folder: {metadata_path}')

    try:
        os.mkdir(metadata_path)
        print(f'new folder created: {metadata_path}')
    except FileExistsError:
        print(f'folder already exists: {metadata_path}')

    for file in files:
        print(f'file: {file}')

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


def process_file(file):
    """
    The process_file function takes a file path as input and returns the image data,
     metadata, and additional metadata.
    The function uses the get_array_czi function from czifile to extract these three
     pieces of information from a .czi file.


    Args:
        file: Specify the file that is being processed

    Returns:
        A tuple of three values: img_data, metadata, and add_metadata
    """

    print(f'processing file: {file}')
    image_data, image_metadata, image_add_metadata = get_array_czi(file, return_addmd=False)
    return image_data, image_metadata, image_add_metadata

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

    czi_files = [file for file in files if file.endswith('.czi')]



    #with concurrent.futures.ThreadPoolExecutor() as executor:
        #results = list(executor.map(process_file, czi_files))

    with multiprocessing.Pool() as pool:
        results = pool.map(process_file, czi_files)

    for img_data, metadata, add_metadata in results:
        all_img_data.append(img_data)
        all_metadata.append(metadata)
        all_add_metadata.append(add_metadata)

    if write_metadata:
        path = os.path.dirname(files[0])
        print(f'path: {path}')
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

    print(f'files: {files}')

    for file in files:
        # get the array and the metadata
        print(f'processing file: {file}')
        img_data, metadata, add_metadata = get_array_czi(file, return_addmd=False)
        all_img_data.append(img_data)
        all_metadata.append(metadata)
        all_add_metadata.append(add_metadata)

    if write_metadata:
        path = os.path.dirname(files[0])
        print(f'path: {path}')
        write_metadata_xml(path, files)

    return all_img_data, all_metadata, all_add_metadata


def extract_channels(img_data: list):
    """
    The extract_channels function extracts the first channel
     of each image in a list of images.

    Args:
        img_data:list: Store the image data

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
