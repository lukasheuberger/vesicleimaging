# Description: This file contains functions that display information about the czi files.

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
