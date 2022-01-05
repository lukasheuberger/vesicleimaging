"""Plotting of czi images and image analysis."""
import format
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import cv2
import numpy as np
import czi_image_handling as handler
import os
from icecream import ic

def plot_images(img_xy_data, img_add_metadata, img_metadata, channels, saving_on=False, scalebar=True):

    format.formatLH()
    for img_index, img in enumerate(img_xy_data):
        ic(img_index)#, img)
        # print ('image:', img)
        # print ('index:', index)
        # zstacks = (img_metadata[img_index][0]['Shape_czifile'][4])

        image = img[0]

        scaling_x = handler.disp_scaling(img_add_metadata[img_index])
        ic(scaling_x)

        print(img_metadata[img_index][0]['Filename'])

        for channel_index, channel_img in enumerate(image): #enumerates channels
            ic(channel_index)
            ic(channels[channel_index])

            for z_index, z_img in enumerate(channel_img):
                #ic(z_index) #plt.imshow(imx) #ic(inx, imx)

                temp_filename = img_metadata[img_index][0]['Filename'].replace('.czi', '')
                title_filename = ''.join([temp_filename, '_', channels[channel_index], '_', str(z_index)])
                output_filename = ''.join(['analysis/', title_filename, '.png'])

                fig = plt.figure(figsize=(5, 5), frameon=False)
                fig.tight_layout(pad=0)
                plt.imshow(z_img, cmap='gray')

                scalebar = ScaleBar(dx=scaling_x[0], location='lower right', fixed_value=30,
                                    fixed_units='µm', frameon=False, color='w')  # 1 pixel = scale [m]
                plt.gca().add_artist(scalebar)
                plt.axis('off')
                plt.title(title_filename)

                if saving_on:
                    plt.savefig(output_filename, dpi=300)  # ,image[channel],cmap='gray')

        # for channel in range(0, len(image)):
        #
        #     for slice in range(0, zstacks):
        #         #pass
        #         ic(image[slice])
        #
        #     fig = plt.figure(figsize=(5, 5), frameon=False)
        #     fig.tight_layout(pad=0)
        #     # subfig_counter = index*3 + channel
        #     plt.imshow(image[channel], cmap='gray')
        #     plt.axis('off')
        #     plt.title(title_filename)
        #
        #     # print ('channel:',channel)
        #     temp_filename = img_metadata[img_index][0]['Filename'].replace('.czi', '')
        #     title_filename = ''.join([temp_filename, '_', channels[channel]])
        #     output_filename = ''.join(['analysis/', title_filename, '.png'])
        #     # print ('index + channel', index*3 + channel + 1)
        #
        #     # print(filename_counter)
        #     # if filename_counter < 1: #so only top two images have channel names
        #     #    axs[subfig_counter].title.set_text(channel_names[channel])
        #     # scalebar = ScaleBar(dx=scaling_x[index], location='lower right', fixed_value=30,
        #     #                    fixed_units='µm', frameon = False, color = 'w')  # 1 pixel = scale [m]
        #     # plt.gca().add_artist(scalebar)
        #
        #     if saving_on:
        #         # print(output_filename)
        #         plt.savefig(output_filename, dpi=300)  # ,image[channel],cmap='gray')
        #     # todo: add scalebar (always)

def detect_circles(img_xy_data, img_metadata, hough_saving, param1_array, param2_array, minmax, display_channel, detection_channel):
    # see https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=houghcircles#houghcircles
    circles = []
    for index, img in enumerate(img_xy_data):
        print ('image',  index + 1, 'is being processed...')
        image = img[0]

        if image.dtype == 'uint16':
            image = handler.convert8bit(image)

        output = image[display_channel].copy()  # output on vis image
        # output = [x + 30 for x in output]
        # output = map(lambda x: x+30, output)
        # output = list(np.asarray(output) + 30)
        # detect circles in the image
        circle = cv2.HoughCircles(image[detection_channel], cv2.HOUGH_GRADIENT,  # detection on vis image
                                  dp=2,
                                  minDist=minmax[1] + 10,
                                  minRadius=minmax[0],
                                  maxRadius=minmax[1],
                                  param1=param1_array[index],
                                  param2=param2_array[index])
        # the bigger param1, the fewer circles may be detected
        # The smaller param2 is, the more false circles may be detected. Circles, corresponding to the larger accumulator values, will be returned first.
        # ensure at least some circles were found
        if circle is not None:
            # convert the (x, y) coordinates and radius of the circles to integers
            circle = np.round(circle[0, :]).astype("int")
            # loop over the (x, y) coordinates and radius of the circles
            for (x, y, r) in circle:
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                # cv2.circle(output, (x, y), (r-30), (255, 255, 255), 2) # x,y,radius
                cv2.circle(output, (x, y), r, (255, 255, 255), 2)  # x,y,radius
                cv2.rectangle(output, (x - 2, y - 2), (x + 2, y + 2), (255, 255, 255), -1)
            # print(circle)
        circles.append(circle)

        fig = plt.figure(figsize=(5, 5), frameon=False)
        fig.tight_layout(pad=0)
        plt.imshow(output, cmap='gray')#, vmin=0, vmax=20)
        plt.axis('off')

        if hough_saving:
            try:
                os.mkdir('analysis/HoughCircles')
            except FileExistsError:
                pass
            temp_filename = img_metadata[index][0]['Filename'].replace('.czi', '')
            output_filename = ''.join(['analysis/HoughCircles/', temp_filename, '_houghcircles.png'])
            # print(output_filename)
            plt.imsave(output_filename, output, cmap='gray')#, vmin=0, vmax=20)

        print ('______________________')

    return circles

def measure_circles(image_xy_data, distance_from_border=20, excel_saving = True):
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
                        dx = x - circle[0]
                        dy = y - circle[1]
                        distanceSquared = dx ** 2 + dy ** 2
                        # print (distanceSquared)
                        if distanceSquared <= (measurement_radius ** 2):
                            # img_five[0][0] = first image, fluorescein channel
                            pixel_val = image[0][dy][dx]  # measurement on GFP image (= 0)
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