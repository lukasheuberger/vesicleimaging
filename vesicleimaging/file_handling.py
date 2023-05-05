import os

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
            # Check if the file has the desired extension and does not contain the exclude keyword
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


#path = '/Volumes/rta-a-scicore-che-palivan$/FG/Palivan/heuber0000/experimental_data/LH23-24/processed'
# Specify your folder path
#files, filenames = find_files(path)
#print(filenames)
