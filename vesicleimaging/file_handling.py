import os
import h5py
import pickle
import numpy as np


def find_files(directory, file_ext='.czi', exclude_keyword='placeholder', append_method='_'):
    """
    This function searches for files in a given directory and its subdirectories,
    filtering them based on the specified file extension and excluding files
    containing a specific keyword. It also generates a list of filenames,
    combining the names of the parent directories.

    :param directory: The directory to search for files.
    :type directory: str
    :param file_ext: The file extension to look for, defaults to '.czi'.
    :type file_ext: str, optional
    :param exclude_keyword: The keyword to exclude files by, defaults to 'placeholder'.
    :type exclude_keyword: str, optional
    :param append_method: The character to use when appending parent directory names, defaults to '_'.
    :type append_method: str, optional
    :return: A tuple containing the list of found files and the list of generated filenames.
    :rtype: tuple
    """

    found_files = []
    filenames = []

    # Iterate through the directory and its subdirectories
    for root, _, files in os.walk(directory):
        for file in files:
            # Check if the file has the desired extension and does not contain exclude keyword
            if file.endswith(file_ext) and not file.endswith(exclude_keyword + file_ext):
                # Add the file path to the found_files list
                found_files.append(os.path.join(root, file))

                # Extract the names of the parent directories
                parent1 = os.path.basename(root)
                parent2 = os.path.basename(os.path.dirname(root))

                # Combine the parent directory names and append them to the filenames list
                filenames.append(f"{parent1}{append_method}{parent2}")
                filenames.append(f"{parent2}_{parent1}")

    return found_files, filenames


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


def save_h5_files(data: list[int],
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
