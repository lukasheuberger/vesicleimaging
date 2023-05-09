



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

    channels = handler.get_channels([img_add_metadata])
    channel_names = []
    for channel in channels[2]:
        if channel is None:
            channel_names.append('T-PMT')
        elif channel == "BODIPY 630/650-X":
            channel_names.append('BODIPY 630-650-X')
        else:
            channel_names.append(channel.replace(" ", ""))

    scaling_x = handler.disp_scaling([img_add_metadata])

    for channel_index, channel_img in enumerate(image_data):
        # ic(channel_index, channel_img.shape)

        for timepoint_index, timepoint in enumerate(channel_img):
            # ic(timepoint_index, timepoint.shape)

            for zstack_index, zstack in enumerate(timepoint):
                # ic(zstack_index, zstack.shape)

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
                    ic(os.getcwd())
                    new_folder_path = os.path.join(os.getcwd(), 'analysis')
                    try:
                        ic(new_folder_path)
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