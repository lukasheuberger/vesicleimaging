# Description: This file contains functions for image analysis.


def process_image(zstack_img, display_channel, detection_channel, minmax, param1, param2):
    z_circles = []
    output_img = display_channel
    #ic(display_channel.shape)
    #output_img = display_channel.copy()#img[display_channel] \ #todo rename, this is not correct
        #[timepoint_index][zstack_index].copy()

    # Apply Gaussian blur to reduce noise
    gray_blurred = cv2.GaussianBlur(zstack_img, (9, 9), 2)
    # plt.imshow(gray_blurred)
    # ic(output_img.shape)
    circle = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT,
                                  dp=2,
                                  minDist=minmax[1],
                                  minRadius=minmax[0],
                                  maxRadius=minmax[1],
                                  param1=param1,
                                  param2=param2)
    # todo make that the drawing is only if it's also plotted

    if circle is not None:
        # convert the (x, y) coords and radius to integers
        circle = np.round(circle[0, :]).astype("int")
        # loop over the (x, y) coords and radius of circles
        for (x_coord, y_coord, radius) in circle:
            # draw circle in output image and draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(output_img, (x_coord, y_coord),
                       radius, (255, 255, 255), 2)  # x,y,radius
            cv2.rectangle(output_img, (x_coord - 5, y_coord - 5),
                          (x_coord + 5, y_coord + 5),
                          (255, 255, 255), -1)
        z_circles.append(circle)

        return circle, output_img, z_circles
    else:
        return None, None, None


def detect_circles(image_data: list,
                   image_metadata: list,
                   param1_array: list,
                   param2_array: list,
                   minmax: list,
                   display_channel: int,
                   detection_channel: int,
                   plot:bool = True,
                   return_image: bool = False,
                   hough_saving: bool = False,
                   debug: bool = False):
    """
    The detect_circles function takes a list of images and returns
    a list of detected circles.
    The circles are returned as three separate lists, one for each channel.
    Each circle is represented by the x-y coordinates and radius in pixels.

    Args:
        image_data:list: list of images
        image_metadata:list: list of image metadata
        param1_array:list: list of detection sensitivity parameter 1
        param2_array:list: list of detection sensitivity parameter 2
        minmax:list: list of minimum and maximum radius of the detected circles
        display_channel:int: channel to display circles on
        detection_channel:int: channel to detect circles on
        hough_saving:bool=False: output image with detected circles on them

    Returns:
        A list of circles
    """
    # this is all a mess and needs to be fixed
    # todo make params so both arrays and ints work
    # todo make user be able to chose between radius in um and pixels
    # todo make auto-optimization to improve found circles
    # see https://docs.opencv.org/
    # 2.4/modules/imgproc/doc/feature_detection.html?highlight=houghcircles#houghcircles
    # todo for zstack: don't always recalculate but use existing positions

    # if circles.all() == [None]:
    print('the bigger param1, the fewer circles may be detected')
    print('the smaller param2, the more false circle sare detected')
    print('circles, corresponding to larger accumulator values'
          ' will be returned first')
    print('-------------------------')
    print(' ')

    if isinstance(image_data, list) is False:
        raise ValueError('image_data must be a list')

    if isinstance(image_metadata, list) is False:
        raise ValueError('image_metadata must be a list')

    # ic(image_data.shape)

    circles = []

    for index, img in enumerate(image_data):
        try:
            filename = image_metadata[index][0]['Filename']
        except KeyError:
            filename = image_metadata[index]['Filename']
        print(f'file {index+1} ({filename}) is being processed...')

        if img.dtype == 'uint16':
            img = handler.convert8bit(img)

        detection_img = img[detection_channel]
        print(f'detection_img.shape: {detection_img.shape}')

        timepoint_circles = []

        for timepoint_index, timepoint_img in enumerate(detection_img):
            # ic(timepoint_index, timepoint_img.shape)

            z_circles = []
            for zstack_index, zstack_img in enumerate(timepoint_img):
                # ic(zstack_index, zstack_img.shape)

                param1 = param1_array[index] if isinstance(param1_array, list) else param1_array
                param2 = param2_array[index] if isinstance(param2_array, list) else param2_array

                output_img = img[display_channel][timepoint_index][zstack_index].copy()
                circle, output_img, _ = process_image(zstack_img, output_img , detection_channel, minmax, param1, param2)

                if circle is not None:

                    z_circles.append(circle)

                    if debug:
                        print(f'z_circles: {z_circles}')

                    if plot:
                        print(f'output_img.shape: {output_img.shape}')

                        fig = plt.figure(figsize=(5, 5), frameon=False)
                        fig.tight_layout(pad=0)
                        plt.imshow(output_img)  # , vmin=0, vmax=20)
                        plt.axis('off')
                        plt.show()
                        plt.close()

                    if hough_saving:
                        try:
                            os.mkdir('analysis/HoughCircles')
                        except FileExistsError:
                            pass
                        except FileNotFoundError:
                            os.mkdir('analysis')
                            os.mkdir('analysis/HoughCircles')
                        # todo check if this works with this
                        #  zero or needs try except to work
                        temp_filename = image_metadata[index][0]\
                            ['Filename'].replace('.czi', '')
                        output_filename = ''.join(['analysis/HoughCircles/',
                                                   temp_filename,
                                                   '_houghcircles.png'])
                        # ic(output_filename)
                        plt.imsave(output_filename, output_img, cmap='gray')
                        # , vmin=0, vmax=20)

            timepoint_circles.append(z_circles)

        circles.append(timepoint_circles)

    # ic(circles)
    if return_image:
        return circles, output_img
    else:
        return circles

def plot_size_histogram(circles: list,
                    add_metadata: list, bins: float = 10):
    # todo make possible to choose to plot general histogram or per condition
    # todo convert to um instead of pixels
    # todo import formatLH

    ic(len(circles[:][:][:][:])) #todo doesnt work

    radii = []

    for outer_list in circles:
        for inner_list in outer_list:
            for arr in inner_list:
                for list in arr:
                    # ic(list)
                    radii.append(list[2])

    scale_factor = float(handler.disp_scaling(add_metadata[0])[0]) # assume it doesn't change during experiment
    ic(scale_factor)

    ic(type(scale_factor))
    ic(type(radii[0]))

    radii = [2 * radius * scale_factor * 10e5 for radius in radii] # x 10^5 for units in um

    ic(len(radii))


    plt.figure()
    _, bins, _ = plt.hist(radii, bins=bins, color='black')
    xmin = min(radii) - 5
    xmax = max(radii) + 5
    plt.xlim(xmin, xmax)
    # plt.text(min(diameters)-4,0.1, r'n = %s'%(len(diameters)))#, fontdict = font)
    plt.xlabel('size [µm]')
    plt.ylabel('counts')
    #plt.minorticks_off()
    plt.show()


    return radii


@njit
def custom_meshgrid(x_min, x_max, y_min, y_max):
    xs = np.empty((y_max - y_min, x_max - x_min), dtype=np.int32)
    ys = np.empty((y_max - y_min, x_max - x_min), dtype=np.int32)

    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            xs[y - y_min, x - x_min] = x
            ys[y - y_min, x - x_min] = y

    return ys, xs

@njit
def calculate_average_per_circle(zstack_img, measurement_circles, distance_from_border):
    average_per_circle = []
    for circle in measurement_circles:

        x_0, y_0, radius_px = circle
        measurement_radius = radius_px - distance_from_border

        x_min, x_max = x_0 - measurement_radius, x_0 + measurement_radius
        y_min, y_max = y_0 - measurement_radius, y_0 + measurement_radius

        ys, xs = custom_meshgrid(x_min, x_max, y_min, y_max)
        dist_squared = (xs - x_0)**2 + (ys - y_0)**2
        mask = (dist_squared <= measurement_radius**2) & (xs >= 0) & (ys >= 0) & (xs < zstack_img.shape[1]) & (ys < zstack_img.shape[0])

        masked_ys = ys.ravel()[mask.ravel()]
        masked_xs = xs.ravel()[mask.ravel()]

        pixels_in_circle = np.empty_like(masked_ys, dtype=zstack_img.dtype)
        for i in range(masked_ys.size):
            pixels_in_circle[i] = zstack_img[masked_ys[i], masked_xs[i]]

        average_per_circle.append(np.mean(pixels_in_circle))

    return average_per_circle, pixels_in_circle


def measure_circle_intensity(image_data: list,
                             image_metadata: list,
                             circles: list,
                             measurement_channel: int,
                             distance_from_border: int = 10,
                             excel_saving: bool = True,
                             filenames: list = None):
    """
    The measure_circle_intensity function takes a list of
    images and circles as input.
    It returns a dataframe with the measured summarized values.

    Args:
        image_data:list: Pass the image data to be analysed
        circles:list: Pass the circles that are detected in the image
        measurement_channel:int: Specify which channel to measure
        distance_from_border:int=10: distance from border for measurements
        excel_saving:bool=True: Save the results to an excel file

    Returns:
        A dataframe with the following columns:
            image: index from list
            timepoint: index from list
            z_level: index from list
            no_GUVs: number of counted GUVs
            average: average over all counted GUVs in an image
            min: mininum intensity of counted GUVs in an image
            max: maximum intensity of counted GUVs in an image
            stdev: standard deviation of all counted GUVs in an image
        pixels_in_circle:

    """

    # todo check if number of guvs counted is equal to number of guvs in circles array

    if isinstance(image_data, list) is False:
        raise ValueError('image_data must be a list')

    results_df = pd.DataFrame(columns=['image', 'timepoint', 'z_level',
                                       'no_GUVs', 'average', 'min',
                                       'max', 'stdev'])

    intensity_per_circle = []

    for index, img in enumerate(image_data):
        #
        # ic(index, img.shape)
        if filenames is not None:
            filename = filenames[index]
        else:
            try:
                filename = image_metadata[index][0]['Filename']
            except KeyError:
                filename = image_metadata[index]['Filename']

        print(f'file {index + 1} ({filename}) is being processed...')

        if img.dtype == 'uint16':
            img = handler.convert8bit(img)

        detection_img = img[measurement_channel]
        # ic(detection_img.shape)

        circles_per_image = []

        for timepoint_index, timepoint_img in enumerate(detection_img):
            # ic(timepoint_index, timepoint_img.shape)

            for zstack_index, zstack_img in enumerate(timepoint_img):
                # print(f'zstack_index: {zstack_index}, zstack_img.shape: {zstack_img.shape}')

                # if circles[index] is not None: # maybe only [index] here

                #print(circles[index][timepoint_index])
                if circles[index][timepoint_index] == []:
                    print('skipping this image')
                else:
                    try:
                        measurement_circles = circles[index]\
                            [timepoint_index][zstack_index]
                        print(f'Number of circles measured in this image {len(measurement_circles)}')
                        # print(measurement_circles)
                        # measurement_radius = radius_px - distance_from_border
                        average_per_circle, pixels_in_circle = calculate_average_per_circle(zstack_img, measurement_circles,
                                                                         distance_from_border)
                        # todo check if this is actually the right value here
                        # average_per_circle = []
                        # pixels_in_circle = []

                        # for circle in measurement_circles:
                        #     # ic(circle)
                        #     x_0 = circle[0]
                        #     y_0 = circle[1]
                        #     radius_px = circle[2]
                        #     # if radius is adapted to um
                        #     # this needs to be changed too...
                        #     # ic(x_0, y_0, radius_px)
                        #
                        #     # make radius slighly smaller so border is not in range
                        #     measurement_radius = radius_px - distance_from_border
                        #     # ic(measurement_radius)
                        #
                        #     # iterate over all pixels in circle
                        #     for x_coord in range(x_0 - measurement_radius,
                        #                          x_0 + measurement_radius):
                        #         for y_coord in range(y_0 - measurement_radius,
                        #                              y_0 + measurement_radius):
                        #             delta_x = x_coord - x_0
                        #             delta_y = y_coord - y_0
                        #
                        #             distance_squared = delta_x ** 2 + delta_y ** 2
                        #             try:
                        #                 if distance_squared <= \
                        #                         (measurement_radius ** 2):
                        #                     pixel_val = zstack_img[y_coord][x_coord]
                        #                     pixels_in_circle.append(pixel_val)
                        #             except IndexError:
                        #                 pass
                        #                 # print('skipping this circle')
                        #                 # todo fix this!
                        #
                        #     average_per_circle.append(np.mean(pixels_in_circle))

                        circles_per_image.append(average_per_circle)

                        # print('no. of GUVs counted: ', len(measurement_circles))
                        # print('number of pixels: ', len(pixels_in_circle))
                        # print('min: ', np.min(pixels_in_circle))
                        # print('max: ', np.max(pixels_in_circle))
                        # print('average: ', np.mean(pixels_in_circle))
                        # print('stdev: ', np.std(pixels_in_circle))
                        # print('--------------------')
                        if filenames is not None: # combine this with the other if
                            filename = filename.replace('.czi', '')
                        else:
                            try:
                                filename = image_metadata[index][0] \
                                    ['Filename'].replace('.czi', '')
                            except KeyError:
                                filename = image_metadata[index] \
                                    ['Filename'].replace('.czi', '')

                        results_df = results_df.append({
                            'filename': filename,
                            'image': index,
                            'timepoint': timepoint_index,
                            'z_level': zstack_index,
                            'no_GUVs': len(measurement_circles),
                            'average': np.mean(pixels_in_circle),
                            'min': np.min(pixels_in_circle),
                            'max': np.max(pixels_in_circle),
                            'stdev': np.std(pixels_in_circle)
                        }, ignore_index=True)
                    except (TypeError, IndexError):
                        print('skipped this image, no circles found')

        intensity_per_circle.append(circles_per_image)
        print('-----------------------------')

    if excel_saving:
        results_df.to_excel('analysis.xlsx')
        print('excel saved')

    return results_df, intensity_per_circle

