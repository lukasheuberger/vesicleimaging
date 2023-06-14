# Description: This file contains functions that perform operations on images.

import os
import cv2
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np
from .image_info import get_channels, disp_scaling
from matplotlib_scalebar.scalebar import ScaleBar
import pandas as pd


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
        value: Increase the brightness of the image

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
    max_proj = []

    print(f'image_data.shape: {image_data.shape}')
    for channel_index, channel_img in enumerate(image_data):

        print(f'channel_index: {channel_index}')
        print(f'channel_img.shape: {channel_img.shape}')
        for timepoint_index, timepoint in enumerate(channel_img):

            print(f'timepoint_index: {timepoint_index}')
            print(f'timepoint.shape: {timepoint.shape}')
            max_proj = np.max(timepoint, axis=0)

            print(f'max_projection.shape: {max_proj.shape}')

        new_image.append(max_proj)
    return new_image


def plot_images(image_data: list,
                img_metadata: list,
                img_add_metadata: list,
                saving: bool = True,
                display: bool = True,
                scalebar: bool = True,
                scalebar_value: int = 50):
    """
    The plot_images function takes a list of images and plots them
    in the same window. The function also takes a list of metadata
    for each image, which is used to label the plot. The function
    can be passed an argument that allows it to save each
    figure as a png file.

    Args:
        image_data:list: image data to plot
        img_metadata:list: image metadata
        img_add_metadata:list: image additional metadata
        saving:bool=True: Save the images to a folder
        display:bool=True: Display the images
        scalebar:bool=True: Add a scalebar to the images
        scalebar_value:int=50: Set the length of the scalebar in µm

    Returns:
        Nothing
    """

    # todo check if in right form or reduce via handler
    # todo make it take single image and also whole
    #  array of images (see other functions for this)
    # todo exception handling if / is in dye name (e.g. bodipy 630/650)

    # dimension order: C, T, Z, Y, X

    channels = get_channels([img_add_metadata])
    channel_names = []
    for channel in channels[2]:
        if channel is None:
            channel_names.append('T-PMT')
        elif channel == "BODIPY 630/650-X":
            channel_names.append('BODIPY 630-650-X')
        else:
            channel_names.append(channel.replace(" ", ""))

    scaling_x = disp_scaling([img_add_metadata])

    for channel_index, channel_img in enumerate(image_data):
        # print(f'channel_index: {channel_index}')
        # print(f'channel_img.shape: {channel_img.shape}')

        for timepoint_index, timepoint in enumerate(channel_img):
            # print(f'timepoint_index: {timepoint_index}')
            # print(f'timepoint.shape: {timepoint.shape}')

            for zstack_index, zstack in enumerate(timepoint):
                # print(f'zstack_index: {zstack_index}')
                # print(f'zstack.shape: {zstack.shape}')

                try:
                    temp_filename = img_metadata['Filename'].replace('.czi', '')
                except TypeError:
                    temp_filename = img_metadata[0]['Filename'].replace('.czi', '')

                title_filename = ''.join([temp_filename, '_',
                                          channel_names[channel_index], '_t',
                                          str(timepoint_index), '_z',
                                          str(zstack_index)])

                fig = plt.figure(figsize=(5, 5), frameon=False)
                fig.tight_layout(pad=0)
                plt.imshow(zstack, cmap='gray')
                plt.title(title_filename)
                plt.axis('off')
                if scalebar:  # 1 pixel = scale [m]
                    scalebar = ScaleBar(dx=scaling_x[0],
                                        location='lower right',
                                        fixed_value=scalebar_value,
                                        fixed_units='µm',
                                        frameon=False, color='w')

                    plt.gca().add_artist(scalebar)

                if saving:
                    print(f'current working directory: {os.getcwd()}')
                    new_folder_path = os.path.join(os.getcwd(), 'analysis')
                    try:
                        print(f'new_folder_path: {new_folder_path}')
                        os.mkdir(new_folder_path)
                        print('created new analysis folder: ', new_folder_path)
                    except (FileExistsError, FileNotFoundError):
                        pass

                    try:
                        output_filename = ''.join([os.getcwd(), '/analysis/',
                                                  title_filename, '.png'])
                    except OSError:
                        os.mkdir(new_folder_path)
                        output_filename = ''.join([os.getcwd(), '/analysis/',
                                                   title_filename, '.png'])

                    plt.savefig(output_filename, dpi=300)
                    plt.close()
                    # ,image[channel],cmap='gray')
                    print('image saved: ', output_filename)
                if display:
                    plt.show()
                    plt.close()

def circles_to_pandas(detected_circles: list, filenames: list, scaling_factor: float):
    circles_df = pd.DataFrame()

    for index, outer in enumerate(detected_circles):
        # print(filenames[index])
        for inner in outer:
            for arr in inner:
                for array in arr:
                    array_list = array.tolist()
                    # print(array_list)
                    temp_df = pd.DataFrame([array_list], columns=['X', 'Y', 'radius_px'])

                    # print(array_list[2])

                    # add column with scaled radius
                    temp_df['radius_um'] = array_list[2] * scaling_factor * 10e5
                    temp_df['diameter_um'] = array_list[2] * scaling_factor * 10e5 * 2

                    # add a new column with the condition
                    temp_df['filename'] = filenames[index]

                    # append the temporary dataframe to the main dataframe
                    circles_df = circles_df.append(temp_df)


    return circles_df

def split_positions(image_data):
    """
    The split_positions function takes a 6-dimensional numpy array and splits it along the first axis.
    The function returns a list of 5-dimensional arrays, each containing data for one position.

    Args:
        image_data: Pass in the image data

    Returns:
        A list of numpy arrays, each containing a single image
    """

    # Check if the input is a numpy array
    if type(image_data) is not np.ndarray:
        raise ValueError('Input data must be a numpy array')

    # Check if the array has the correct number of dimensions
    if image_data.ndim != 6:
        raise ValueError('Input data must be a 6-dimensional array')

    # Split the array along the first axis
    split_data = np.array_split(image_data, image_data.shape[0])

    # Remove dimensions of size 1
    split_data = [np.squeeze(sub_array, axis=0) for sub_array in split_data]

    return split_data

