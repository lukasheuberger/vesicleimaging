"""
Description: This file contains functions for analyzing images.
"""

import os
from numba import njit
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .image_operations import convert8bit, disp_scaling


def process_image(
    zstack_img, output_img, minmax, param1, param2, plot, hough_saving=False
):
    """
    The process_image function takes in a zstack image, an output image,
    the channel to detect on (0-2), the min and max radius of the circles
    to detect, and two parameters for HoughCircles.
    It then applies a Gaussian blur to reduce noise in the input image.
    Then it uses OpenCV's HoughCircles function to find circles with radii between
    minRadius and maxRadius using param 1 and param 2 as parameters for that function.
    If plot is True it will draw those circles onto the output_img array.

    Args:
        zstack_img: Pass in the image that will be processed
        output_img: Draw the circles on the image
        minmax: Set the minimum and maximum radius of the circles to be detected
        param1: Set the threshold for the canny edge detector
        param2: Set the threshold for the circle detection
        plot: Determine if the output image should be plotted
        hough_saving: Set if hough_circle images should be saved

    Returns:
        The circles, the output image and a list of all circles
    """

    # Apply Gaussian blur to reduce noise
    gray_blurred = cv2.GaussianBlur(zstack_img, (9, 9), 1.5)
    # plt.imshow(gray_blurred)
    # print(f'output_img.shape:{output_img.shape}')
    circle = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT_ALT,  # alternative: HOUGH_GRADIENT
        dp=1.5,
        minDist=minmax[1] / 2,
        minRadius=minmax[0],
        maxRadius=minmax[1],
        param1=param1,
        param2=param2,
    )
    # print(circle)
    if circle is not None:
        # Round off the (x, y) coordinates and radius to integers
        circle = np.round(circle[0, :]).astype(int)

        if plot is True or hough_saving is True:
            for (x, y, radius) in circle:
                # Draw the circle on the output image
                cv2.circle(output_img, (x, y), radius, (255, 255, 255), 2)

                # Draw a centered rectangle at the center of the circle
                cv2.rectangle(
                    output_img, (x - 5, y - 5), (x + 5, y + 5), (255, 255, 255), -1
                )

        return circle, output_img
    else:
        return None, None


def detect_circles(
    image_data: list,
    image_metadata: list,
    param1_array: list,
    param2_array: list,
    minmax: list,
    display_channel: int,
    detection_channel: int,
    plot: bool = True,
    return_image: bool = False,
    hough_saving: bool = False,
    debug: bool = False,
):
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
        plot:bool=True: plot detected circles
        return_image:bool=False: return image with detected circles on them
        hough_saving:bool=False: output image with detected circles on them
        debug:bool=False: print debug messages

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

    print("@param1: threshold for the canny edge detector, around 300 works well")
    print("@param2: circle 'perfectness' measure. The closer it to 1, the better "
          "shaped circles algorithm selects. In most cases 0.9 should be fine")
    print("-------------------------")
    print(" ")

    if isinstance(image_data, list) is False:
        raise ValueError("image_data must be a list")

    if isinstance(image_metadata, list) is False:
        raise ValueError("image_metadata must be a list")

    circles = []
    output_img = None

    args = (
        image_metadata,
        param1_array,
        param2_array,
        display_channel,
        minmax,
        plot,
        hough_saving,
        debug,
        detection_channel,
    )

    for index, img in enumerate(image_data):
        try:
            filename = image_metadata[index][0]["Filename"]
        except KeyError:
            filename = image_metadata[index]["Filename"]
        print(f"file {index + 1} ({filename}) is being processed...")

        if img.dtype == "uint16":
            img = convert8bit(img)

        # print(f'img.shape: {img.shape}')

        if len(img.shape) == 5:
            print("5dim, only one position")

            # detection_img = img[detection_channel]
            # print(f'detection_img.shape: {detection_img.shape}')
            # directly iterate through timepoints, zstack

            timepoint_circles = iterate_circles(*args, index, img)
            # print([timepoint_circles])

            circles.append([timepoint_circles])
            # put in brackets to make same form as if it has positions, just with a one first
        elif len(img.shape) == 6:
            print("6dim, multiple positions")
            print(f"img.shape:{img.shape}")
            # iterate through positions first, then timepoints & zstack

            position_circles = []
            for position_index, position in enumerate(img):
                # print(f'position_index: {position_index}, position.shape: {position.shape}')
                timepoint_circles = iterate_circles(
                    *args, index, position, position_index
                )
                # print(timepoint_circles)
                position_circles.append(timepoint_circles)
                # print(position_circles)
                print("---------- next position")
            circles.append(position_circles)
        else:
            print("unknown dim, please check your data")
            break
        print("-------------------------")

    if return_image:
        return circles, output_img
    else:
        return circles


def iterate_circles(
    image_metadata,
    param1_array,
    param2_array,
    display_channel,
    minmax,
    plot,
    hough_saving,
    debug,
    detection_channel,
    index,
    img,
    position_index=0,
):
    """
    The iterate_circles function iterates through the images in a list of filenames,
        and returns a list of circles detected by HoughCircles.

    Args:
        index: Access the correct element in param_array
        img: Display the image in the plot
        image_metadata: Get the filename of the image
        param1_array: Set the parameter for the houghcircles function
        param2_array: Set the threshold for the houghcircles function
        display_channel: Select which channel to display the circles on
        minmax: Set the minimum and maximum values for the
        plot: Plot the image with the detected circles
        hough_saving: Save the images with the detected circles
        debug: Print the circles detected in each z-stack
        detection_channel: channel index where circles should be detected
        position_index: Select the position index of a multi-position image

    Returns:
        A list of lists of circles (x, y, r)
    """

    detection_img = img[detection_channel]

    timepoint_circles = []
    for timepoint_index, timepoint_img in enumerate(detection_img):
        # print(f'timepoint_index: {timepoint_index}, timepoint_img.shape: {timepoint_img.shape}')

        z_circles = []
        for zstack_index, zstack_img in enumerate(timepoint_img):
            # print(f'zstack_index: {zstack_index}, zstack_img.shape: {zstack_img.shape}')

            param1 = (
                param1_array[index] if isinstance(param1_array, list) else param1_array
            )
            param2 = (
                param2_array[index] if isinstance(param2_array, list) else param2_array
            )

            if len(img.shape) == 5:
                output_img = img[display_channel][timepoint_index][zstack_index].copy()
            elif len(img.shape) == 6:
                output_img = img[position_index][display_channel][timepoint_index][
                    zstack_index
                ].copy()

            # print(f'output_img.shape: {output_img.shape}')

            circle, output_img = process_image(
                zstack_img, output_img, minmax, param1, param2, plot, hough_saving
            )
            # print(circle)
            # print(f'output_img.shape: {output_img.shape}')

            # try:
            #     print(f'output_img.shape: {output_img.shape}')
            # except AttributeError:
            #     print(f'no circles detected on {filenames[index]}')

            # print(f'{len(circle)} circles found')

            if circle is not None:
                print(f"{len(circle)} circle(s) found")
                z_circles.append(circle)

                if debug:
                    print(f"z_circles: {z_circles}")

                if plot:
                    # print(f'output_img.shape: {output_img.shape}')

                    fig = plt.figure(figsize=(5, 5), frameon=False)
                    fig.tight_layout(pad=0)
                    plt.imshow(output_img)  # , vmin=0, vmax=20)
                    plt.axis("off")
                    plt.show()
                    plt.close()

                if hough_saving:
                    # todo change file naming so it doesn't overwrite constantly
                    try:
                        os.mkdir("analysis/HoughCircles")
                    except FileExistsError:
                        pass
                    except FileNotFoundError:
                        os.mkdir("analysis")
                        os.mkdir("analysis/HoughCircles")

                    # todo check if this works with this
                    # zero or needs try except to work
                    temp_filename = image_metadata[index][0]["Filename"].replace(
                        ".czi", ""
                    )
                    output_filename = "".join(
                        ["analysis/HoughCircles/", temp_filename, "_houghcircles.png"]
                    )
                    # print(f'output_filename: {output_filename}')
                    fig = plt.figure(figsize=(5, 5), frameon=False)
                    fig.tight_layout(pad=0)
                    plt.imshow(output_img)  # , vmin=0, vmax=20)
                    plt.axis("off")
                    plt.imsave(output_filename, output_img, cmap="gray")
                    plt.close()
            else:
                print("no circles detected")

        timepoint_circles.append(z_circles)
    return timepoint_circles


def plot_size_histogram(circles: list, add_metadata: list, bins: float = 10):
    # todo make possible to choose to plot general histogram or per condition
    # todo convert to um instead of pixels
    # todo import formatLH

    # ic(len(circles[:][:][:][:])) #todo doesnt work

    """
    The plot_size_histogram function takes a list of circles and plots the size distribution.

    Args:
        circles: list: Pass the list of circles to the function
        add_metadata: list: Get the scaling factor from the metadata
        bins: float: Set the number of bins in the histogram

    Returns:
        A list of the radii of all circles in the image
    """

    radii = []

    for outer in circles:
        for inner in outer:
            for arr in inner:
                for li in arr:
                    radii.append(li[2])

    scale_factor = float(
        disp_scaling(add_metadata[0])[0]
    )  # assume it doesn't change during experiment
    print(f"scale_factor: {scale_factor}")

    print(f"type(scale_factor): {type(scale_factor)}")
    print(f"type(radii[0]): {type(radii[0])}")

    radii = [
        2 * radius * scale_factor * 10e5 for radius in radii
    ]  # x 10^5 for units in um

    print(f"len(radii): {len(radii)}")

    plt.figure()
    _, bins, _ = plt.hist(radii, bins=bins, color="black")
    xmin = min(radii) - 5
    xmax = max(radii) + 5
    plt.xlim(xmin, xmax)
    # plt.text(min(diameters)-4,0.1, r'n = %s'%(len(diameters)))#, fontdict = font)
    plt.xlabel("size [µm]")
    plt.ylabel("counts")
    # plt.minorticks_off()
    plt.show()

    return radii


@njit
def custom_meshgrid(x_min, x_max, y_min, y_max):
    """
    The custom_meshgrid function takes in four arguments: x_min, x_max, y_min and y_max.
    It returns two arrays of the same shape as the input image (i.e., height by width).
    The first array contains all the x-coordinates for each pixel in the image;
    the second array contains all the y-coordinates for each pixel in the image.

    Args:
        x_min: Set the minimum value of x in the grid
        x_max: Determine the width of the grid
        y_min: Set the starting y value of the meshgrid
        y_max: Set the maximum y value of the grid

    Returns:
        A meshgrid of the specified size
    """

    xs = np.empty((y_max - y_min, x_max - x_min), dtype=np.int32)
    ys = np.empty((y_max - y_min, x_max - x_min), dtype=np.int32)

    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            xs[y - y_min, x - x_min] = x
            ys[y - y_min, x - x_min] = y

    return ys, xs


@njit
def calculate_average_per_circle(zstack_img, measurement_circles, distance_from_border):
    """
    The calculate_average_per_circle function takes a zstack image and a list of circles,
    and returns the average pixel intensity per circle. The function also returns an array
    of all pixels in each circle.

    Args:
        zstack_img: Pass the image to the function
        measurement_circles: Get the coordinates of the circles
        distance_from_border: Determine how far away from the edge of the circle

    Returns:
        The average value of all pixels in the circle, and an array of all pixels in the circle
    """

    # print(f'zstack_img.shape: {zstack_img.shape}') # uncommenting this breaks numba
    # zstack_img=zstack_img[0]
    # print(f'zstack_img.shape: {zstack_img.shape}')

    average_per_circle = []
    # pixels_in_circle_list = []
    # print(measurement_circles)

    for circle in measurement_circles:
        # print(f'circle: {circle}')

        x_0, y_0, radius_px = circle
        measurement_radius = radius_px - distance_from_border

        x_min, x_max = x_0 - measurement_radius, x_0 + measurement_radius
        y_min, y_max = y_0 - measurement_radius, y_0 + measurement_radius

        ys, xs = custom_meshgrid(x_min, x_max, y_min, y_max)
        # print(f'ys: {ys}, xs: {xs}')

        dist_squared = (xs - x_0) ** 2 + (ys - y_0) ** 2
        mask = (
            (dist_squared <= measurement_radius**2)
            & (xs >= 0)
            & (ys >= 0)
            & (xs < zstack_img.shape[1])
            & (ys < zstack_img.shape[0])
        )
        # print(f'mask: {mask}')
        # plt.imshow(mask)

        masked_ys = ys.ravel()[mask.ravel()]
        masked_xs = xs.ravel()[mask.ravel()]

        # print(f'masked_ys: {masked_ys}')
        # print(f'masked_xs: {masked_xs}')

        pixels_in_circle_array = np.empty_like(
            masked_ys, dtype=zstack_img.dtype
        )  # Renamed variable
        # pixels_in_circle = np.empty_like(masked_ys, dtype=zstack_img.dtype)
        for i in range(masked_ys.size):
            pixels_in_circle_array[i] = zstack_img[masked_ys[i], masked_xs[i]]
            # pixels_in_circle[i] = zstack_img[masked_ys[i], masked_xs[i]]

        # average_per_circle.append(np.mean(pixels_in_circle))
        average_per_circle.append(np.mean(pixels_in_circle_array))

    return average_per_circle, pixels_in_circle_array


def measure_circle_intensity(
    image_data: list,
    image_metadata: list,
    circles: list,
    measurement_channel: int,
    distance_from_border: int = 10,
    excel_saving: bool = True,
    filenames: list = None,
):
    """
    The measure_circle_intensity function takes a list of
    images and circles as input.
    It returns a dataframe with the measured summarized values.

    Args:
        image_data:list: Pass the image data to be analysed
        image_metadata:list: Pass the image metadata to be analysed
        circles:list: Pass the circles that are detected in the image
        measurement_channel:int: Specify which channel to measure
        distance_from_border:int=10: distance from border for measurements
        excel_saving:bool=True: Save the results to an Excel file
        filenames:list=None: Pass the filenames to be saved to excel

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

    """

    if isinstance(image_data, list) is False:
        raise ValueError("image_data must be a list")

    results_df = pd.DataFrame(
        columns=[
            "image",
            "timepoint",
            "z_level",
            "no_GUVs",
            "average",
            "min",
            "max",
            "stdev",
        ]
    )

    for index, img in enumerate(image_data):
        if filenames is None:
            try:
                filename = image_metadata[index][0]["Filename"]
            except KeyError:
                filename = image_metadata[index]["Filename"]
        else:
            filename = filenames[index]

        print(f"file {index + 1} ({filename}) is being processed...")

        if img.dtype == "uint16":
            img = convert8bit(img)

        dim = len(img.shape)

        if dim == 5:
            print("5dim, only one position")

            detection_img = img[measurement_channel]
            # print(f'detection_img.shape: {detection_img.shape}')

            position_df = iterate_measure(
                detection_img,
                image_metadata[index],
                dim,
                circles,
                index,
                distance_from_border,
                filename
            )

            results_df = pd.concat([results_df, position_df])

        elif dim == 6:
            print("6dim, multiple positions")

            for position_index, position_img in enumerate(img):
                # print(f'position_index: {position_index},
                # position_img.shape: {position_img.shape}')
                detection_img = position_img[measurement_channel]

                position_df = iterate_measure(
                    detection_img,
                    image_metadata[index],
                    dim,
                    circles,
                    index,
                    distance_from_border,
                    filename,
                    position_index,
                )
                results_df = pd.concat([results_df, position_df])

        else:
            print("not 5 or 6 dim, fix your image input")
            break

    if excel_saving:
        results_df.to_excel("analysis.xlsx")
        print("excel saved")

    return results_df  # , intensity_per_circle


def iterate_measure(
    img: list,
    img_metadata: list,
    dim: int,
    circles: list,
    index: int,
    distance_from_border: int,
    filename: str,
    position_index: int = 0,
):
    """
    The iterate_measure function takes a list of images and circles as input.
        Args:
            img: A list of images, each image is a timepoint with z-stacks.
            dim: The dimensionality of the data (5 or 6). 5 for 2D data, 6 for 3D data.
            circles: A list containing lists with all the detected GUVs in an image.
                Each element in this outermost list corresponds to one
                image/position/timepoint combination (i.e., one .czi file).

    Args:
        img: object: Pass the image data to the function
        dim: object: Determine the dimension of the circles list
        circles: object: Pass the circles to the function
        index: object: Specify which image in the list of images to use
        distance_from_border: object: Define the distance from the border of a circle to measure
        filename: object: Get the filename of the image
        position_index: object: Select the circles for a specific position

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
    """

    intensity_per_circle = []
    circles_per_image = []


    position_df = pd.DataFrame(
        columns=[
            "image",
            "timepoint",
            "z_level",
            "no_GUVs",
            "average",
            "min",
            "max",
            "stdev",
        ]
    )

    for timepoint_index, timepoint_img in enumerate(img):
        # print(f'timepoint_index: {timepoint_index}, timepoint_img.shape: {timepoint_img.shape}')

        for zstack_index, zstack_img in enumerate(timepoint_img):
            # print(f'zstack_index: {zstack_index}, zstack_img.shape: {zstack_img.shape}')

            # if not circles[index][position_index][timepoint_index]:
            if not circles[index][0][timepoint_index]:
                print("skipping this image")
            else:
                try:
                    if dim == 5:
                        measurement_circles = circles[index][0][timepoint_index][
                            zstack_index
                        ]
                    elif dim == 6:
                        measurement_circles = circles[index][position_index][
                            timepoint_index
                        ][zstack_index]
                    # print(f'measurement_circles: {measurement_circles}')

                    print(
                        f"Number of circles measured in this image: {len(measurement_circles)}"
                    )

                    # measurement_radius = radius_px - distance_from_border

                    average_per_circle, pixels_in_circle = calculate_average_per_circle(
                        zstack_img, measurement_circles, distance_from_border
                    )

                    circles_per_image.append(average_per_circle)
                    # print(f'average_per_circle: {average_per_circle}')

                    if filename is not None:  # combine this with the other if possible
                        filename = filename.replace(".czi", "")
                    else:
                        try:
                            filename = image_metadata[index][0]["Filename"].replace(
                                ".czi", ""
                            )
                        except KeyError:
                            filename = image_metadata[index]["Filename"].replace(
                                ".czi", ""
                            )

                    position_df = position_df.append(
                        {
                            "filename": filename,
                            "image": index,
                            "position": int(position_index),
                            "timepoint": timepoint_index,
                            "z_level": zstack_index,
                            "no_GUVs": len(measurement_circles),
                            "average": np.mean(pixels_in_circle),
                            "min": np.min(pixels_in_circle),
                            "max": np.max(pixels_in_circle),
                            "stdev": np.std(pixels_in_circle),
                        },
                        ignore_index=True,
                    )

                except (TypeError, IndexError):
                    print("skipped this image, no circles found")

        intensity_per_circle.append(circles_per_image)
        print("-----------------------------")
    return position_df


def detect_circles_tif(
    image_data,
    filenames,
    param1_array,
    param2_array,
    minmax,
    plot=False,
    hough_saving=False,
    debug=False,
):

    """
    The detect_circles_tif function takes in a list of image data, filenames,
    param_array (a list of parameters for the HoughCircles function),
    minmax (the minimum and maximum radius to search for circles), plot (whether or not to show
    plots as it runs through the images) and hough_saving (whether or not to save plots).
    It returns a list of circles.

    Args:
        image_data: Pass the image data to the function
        filenames: Save the images with the circles detected
        param1_array: Specify the upper threshold for the canny edge detector
        param2_array: Set the threshold for the hough transform
        minmax: Set the minimum and maximum values for the image
        plot: Plot the image with the circle detected
        hough_saving: Save the images with the circles detected
        debug: Print the circles found in each image

    Returns:
        A list of circles
    """

    circles = []

    for index, image in enumerate(image_data):
        print(index)
        filename = filenames[index]
        print(f"file {index + 1} ({filename}) is being processed...")

        param1 = param1_array[index] if isinstance(param1_array, list) else param1_array
        param2 = param2_array[index] if isinstance(param2_array, list) else param2_array

        output_img = image.copy()
        circle, output_img = process_image(
            image, output_img, minmax, param1, param2, plot
        )

        if circle is not None:
            circles.append(circle)
        else:
            circles.append([])

        if debug:
            print(f"z_circles: {circles}")

        if plot:
            # print(f'output_img.shape: {output_img.shape}')
            # todo fix that it also put the circles on the plotted image

            fig = plt.figure(figsize=(5, 5), frameon=False)
            fig.tight_layout(pad=0)
            plt.imshow(output_img)  # , vmin=0, vmax=20)
            plt.axis("off")
            plt.show()
            plt.close()

        if hough_saving:
            try:
                os.mkdir("analysis/HoughCircles")
            except FileExistsError:
                pass
            except FileNotFoundError:
                os.mkdir("analysis")
                os.mkdir("analysis/HoughCircles")

            # todo check if this works with this
            # zero or needs try except to work
            output_filename = "".join(
                ["analysis/HoughCircles/", filename, "_houghcircles.png"]
            )
            # print(f'output_filename: {output_filename}')
            fig = plt.figure(figsize=(5, 5), frameon=False)
            fig.tight_layout(pad=0)
            plt.imshow(output_img)  # , vmin=0, vmax=20)
            plt.axis("off")
            plt.imsave(output_filename, output_img, cmap="gray")
            plt.close()

    return circles
