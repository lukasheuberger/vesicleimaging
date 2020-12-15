"""Plotting of czi images and image analysis."""

from vesicle_imaging import format
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import cv2
import numpy as np
from skimage import img_as_ubyte
from vesicle_imaging import czi_image_handling as img_handler


def plot_images(img_xy_data, img_add_metadata, img_metadata, saving_on, channels):
    scaling_x = img_handler.disp_scaling(img_add_metadata)
    format.formatLH()
    for index, img in enumerate(img_xy_data):
        # print ('image:', img)
        image = img[0]
        print(img_metadata[index][0]['Filename'])
        for channel in range(0, len(image)):
            # print ('channel:',channel)
            temp_filename = img_metadata[index][0]['Filename'].replace('.czi', '')
            output_filename = ''.join(['analysis/', temp_filename, '_', channels[channel], '.png'])
            # print ('index + channel', index*3 + channel + 1)
            fig = plt.figure(figsize=(5, 5), frameon=False)
            fig.tight_layout(pad=0)
            # subfig_counter = index*3 + channel
            plt.imshow(image[channel], cmap='gray')
            plt.axis('off')
            plt.title(output_filename)
            # print(filename_counter)
            # if filename_counter < 1: #so only top two images have channel names
            #    axs[subfig_counter].title.set_text(channel_names[channel])
            scalebar = ScaleBar(dx=scaling_x[index], location='lower right', fixed_value=30,
                                fixed_units='µm', frameon = False, color = 'w')  # 1 pixel = scale [m]
            plt.gca().add_artist(scalebar)

            if saving_on == True:
                # print(output_filename)
                plt.savefig(output_filename, dpi=300)  # ,image[channel],cmap='gray')

def detect_circles(img_xy_data, img_metadata, hough_saving, param1_array, param2_array, display_channel, detection_channel):
    # see https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=houghcircles#houghcircles
    circles = []
    for index, img in enumerate(img_xy_data):
        print ('image',  index + 1, 'is being processed...')
        image = img[0]

        if image.dtype == 'uint16':
            print ('image is uint16, converting to uint8 ...')
            image = img_as_ubyte(image)
            print ('done converting to uint8')

        output = image[display_channel].copy()  # output on vis image
        # detect circles in the image
        circle = cv2.HoughCircles(image[detection_channel], cv2.HOUGH_GRADIENT,  # detection on vis image
                                  dp=2,
                                  minDist=100,
                                  minRadius=100,
                                  maxRadius=130, # todo: include min and max radius in input
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
        plt.imshow(output, cmap='gray')
        plt.axis('off')

        if hough_saving == True:
            temp_filename = img_metadata[index][0]['Filename'].replace('.czi', '')
            output_filename = ''.join(['analysis/', temp_filename, '_houghcircles.png'])
            # print(output_filename)
            plt.imsave(output_filename, output, cmap='hot')

    return circles
