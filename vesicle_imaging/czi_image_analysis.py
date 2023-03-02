"""Plotting of czi images and image analysis."""
import czi_image_handling as handler

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from icecream import ic
from matplotlib_scalebar.scalebar import ScaleBar

import format


def plot_images(image_data, img_metadata, img_add_metadata, saving=True, scalebar=True, scalebar_value=50):
    # todo check if in right form or reduce via handler
    # todo make channel name automatic

    # dimension order: C, T, Z, Y, X

    channels = handler.get_channels([img_add_metadata])
    channel_names = []
    for channel in channels[2]:
        if channel == None:
            channel_names.append('T-PMT')
        else:
            channel_names.append(channel.replace(" ", ""))

    scaling_x = handler.disp_scaling([img_add_metadata])

    for channel_index, channel_img in enumerate(image_data):
        # ic(channel_index, channel_img.shape)

        for timepoint_index, timepoint in enumerate(channel_img):
            # ic(timepoint_index, timepoint.shape)

            for zstack_index, zstack in enumerate(timepoint):
                # ic(zstack_index, zstack.shape)

                temp_filename = img_metadata['Filename'].replace('.czi', '')
                title_filename = ''.join([temp_filename, '_', channel_names[channel_index], '_t', str(timepoint_index), '_z', str(zstack_index)])
                output_filename = ''.join(['analysis/', title_filename, '.png'])

                format.formatLH()
                fig = plt.figure(figsize=(5, 5), frameon=False)
                fig.tight_layout(pad=0)
                plt.imshow(zstack, cmap='gray')
                plt.title(title_filename)
                plt.axis('off')
                scalebar = ScaleBar(dx=scaling_x[0], location='lower right', fixed_value=scalebar_value, fixed_units='µm', frameon=False, color='w')  # 1 pixel = scale [m]
                plt.gca().add_artist(scalebar)

                if saving:
                    try:
                        new_folder_path = os.path.join(os.getcwd(), 'analysis')
                        os.mkdir(new_folder_path)
                        print('created new analysis folder: ', new_folder_path)
                    except FileExistsError:
                        pass
                    plt.savefig(output_filename, dpi=300)  # ,image[channel],cmap='gray')
                    print('image saved: ', output_filename)

                plt.show()


def detect_circles(image_data, image_metadata, hough_saving, param1_array, param2_array, minmax, display_channel,
                   detection_channel):
    # this is all a mess and needs to be fixed
    # todo make params so both arrays and ints work
    # todo make user be able to chose between radius in um and pixels
    # see https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=houghcircles#houghcircles

    # if circles.all() == [None]:
    print('the bigger param1, the fewer circles may be detected')
    print('the smaller param2 is, the more false circles may be detected')
    print('circles, corresponding to the larger accumulator values, will be returned first')
    print('-------------------------')
    print(' ')

    if isinstance(image_data, list) is False:
        raise ValueError('image_data must be a list')

    if isinstance(image_metadata, list) is False:
        raise ValueError('image_metadata must be a list')


    # ic(image_data.shape)

    circles = []

    for index, img in enumerate(image_data):
        print('image', index + 1, 'is being processed...')

        if img.dtype == 'uint16':
            img = handler.convert8bit(img)

        detection_img = img[detection_channel]
        #ic(detection_img.shape)

        timepoint_circles = []

        for timepoint_index, timepoint_img in enumerate(detection_img):
            #ic(timepoint_index, timepoint_img.shape)

            z_circles = []
            for zstack_index, zstack_img in enumerate(timepoint_img):
                #ic(zstack_index, zstack_img.shape)

                output_img = img[display_channel][timepoint_index][zstack_index]
                #ic(output_img.shape)

                circle = cv2.HoughCircles(zstack_img, cv2.HOUGH_GRADIENT,
                                          dp=2,
                                          minDist=minmax[1],
                                          minRadius=minmax[0],
                                          maxRadius=minmax[1],
                                          param1=param1_array[index],
                                          param2=param2_array[index])

                if circle is not None:
                    # convert the (x, y) coordinates and radius of the circles to integers
                    circle = np.round(circle[0, :]).astype("int")
                    # loop over the (x, y) coordinates and radius of the circles
                    for (x, y, r) in circle:
                        # draw the circle in the output image, then draw a rectangle
                        # corresponding to the center of the circle
                        cv2.circle(output_img, (x, y), r, (255, 255, 255), 2)  # x,y,radius
                        cv2.rectangle(output_img, (x - 2, y - 2), (x + 2, y + 2), (255, 255, 255), -1)
                    z_circles.append(circle)

                fig = plt.figure(figsize=(5, 5), frameon=False)
                fig.tight_layout(pad=0)
                plt.imshow(output_img)  # , vmin=0, vmax=20)
                plt.axis('off')
                plt.show()

                if hough_saving:
                    try:
                        os.mkdir('analysis/HoughCircles')
                    except FileExistsError:
                        pass

                    temp_filename = image_metadata[index]['Filename'].replace('.czi', '')
                    output_filename = ''.join(['analysis/HoughCircles/', temp_filename, '_houghcircles.png'])
                    # print(output_filename)
                    plt.imsave(output_filename, output_img, cmap='gray')  # , vmin=0, vmax=20)

                #print('______________________')

            timepoint_circles.append(z_circles)

        circles.append(timepoint_circles)

    ic(circles)
    return circles


def measure_circle_intensity(image_xy_data, distance_from_border=10, excel_saving=True):
    img_counter = 0
    average_per_img = []

    for index, img in enumerate(image_xy_data):
        image = img[0]
        # plt.imshow(image[0])
        output = image[1].copy()  # output on vis image
        pixels_in_circle = []
        if detected_circles[index] is not None:
            for circle in detected_circles[index]:

                # print (circle)
                # test_circle = [ 163, 1487,  108] # x,y,radius
                x_0 = circle[0]
                y_0 = circle[1]
                radius = circle[2]

                # make radius slighly smaller so border is not in range
                measurement_radius = radius - distance_from_border  # [index]
                # print (measurement_radius)

                for x in range(x_0 - measurement_radius, x_0 + measurement_radius):
                    for y in range(y_0 - measurement_radius, y_0 + measurement_radius):
                        # todo this might be wrong, see czi_frap

                        # iterate over all pixels in bleached circle
                        for x_coord in range(x_0 - radius_px, x_0 + radius_px):
                            for y_coord in range(y_0 - radius_px, y_0 + radius_px):
                                delta_x = x_coord - x_0
                                delta_y = y_coord - y_0

                                distance_squared = delta_x ** 2 + delta_y ** 2
                                if distance_squared <= (radius_px ** 2):
                                    pixel_val = measurement_image[0][y_coord][x_coord]
                                    pixels_in_circle.append(pixel_val)


                        dx = x - circle[0]
                        dy = y - circle[1]
                        distanceSquared = dx ** 2 + dy ** 2
                        # print (distanceSquared)
                        if distanceSquared <= (measurement_radius ** 2):
                            # img_five[0][0] = first image, fluorescein channel
                            pixel_val = image[0][dy][dx]  # measurement on GFP image (= 0)
                            # todo this should probably be y, x instead of dy, dx
                            pixels_in_circle.append(pixel_val)


                cv2.circle(output, (x_0, y_0), (measurement_radius), (255, 255, 255), 2)  # x,y,radius
            print('filename: ', filenames[index])
            print('no. of GUVs counted: ', len(detected_circles[index]))
            print('number of pixels: ', len(pixels_in_circle))
            print('min: ', np.min(pixels_in_circle))
            print('max: ', np.max(pixels_in_circle))
            print('average: ', np.mean(pixels_in_circle))
            print('stdev: ', np.std(pixels_in_circle))
            print('--------------------')
        else:
            print('no circles were detected')
        img_average = np.average(pixels_in_circle)
        img_stdev = np.std(pixels_in_circle)

        # print (img_average, img_stdev)
        average_per_img.append(img_average)  # <, img_stdev])

        fig = plt.figure(figsize=(10, 10), frameon=False)
        fig.tight_layout(pad=0)
        plt.imshow(output, cmap='gray')
        plt.axis('off')
        # plt.imsave('output.png',output,cmap='hot')

        print(average_per_img)



def test_all_functions(path):

    from czi_image_handling import load_h5_data, disp_basic_img_info
    os.chdir(path)
    path = os.getcwd()
    ic(path)

    image_data, metadata, add_metadata = load_h5_data(path)
    disp_basic_img_info(image_data, metadata)
    test_index = 0

    # C, T, Z, Y, X
    # ic| image_data[0].shape: (2, 1, 1, 1024, 1024)
    # ic| image_data[1].shape: (2, 1, 52, 1024, 1024)
    # ic| image_data[2].shape: (3, 69, 1, 1024, 1024)

    #plot_images(image_data[test_index], metadata[test_index], add_metadata[test_index], saving = True)

    hough_saving = False
    param1_array = [10]
    param2_array = [150]
    minmax = [30,60]
    display_channel = 0
    detection_channel = 0

    test_data = [image_data[test_index]]
    test_metadata = [metadata[test_index]]

    detected_circles = detect_circles(test_data, test_metadata, add_metadata[test_index],
                                      param1_array = param1_array, param2_array = param2_array, minmax = minmax,
                                      display_channel = display_channel, detection_channel=detection_channel)

    measure_circle_intensity(test_data)

if __name__ == '__main__':
    # path = input('path to data folder: ')
    DATA_PATH = '../test_data/general'

    test_all_functions(DATA_PATH)
# todo make that this is all a class and give the variables to the image instance
# todo make consistent img vs image