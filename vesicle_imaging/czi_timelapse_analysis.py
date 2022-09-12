import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage import img_as_ubyte
from icecream import ic
import czi_image_handling as handler

def detect_params(detect_img, distance_from_border = 5, timepoint = 5, dp = 2, minDist = 10, minRadius = 5, maxRadius = 50, param1top = 50, param2top = 300):
    # todo: include min and max radius in input from image data
    # distance_from_border = 50  # in px
    # param1_range = range(param1top, 5, -5)
    # param2_range = range(param2top, 150, 10)

    image = detect_img[timepoint][0]
    ic(image)
    plt.figure(figsize=(5, 5))
    plt.axis('off')
    plt.imshow(image, cmap = 'gray')
    output = image.copy()  # output on vis image
    # detect circles in the image

    circle_detected = False

    if image.dtype == 'uint16':
        image = handler.convert8bit(image)

    for param2 in range(param2top, 150, -5):
        print (param2)
        for param1 in range(param1top, 1, -2):
            print (param1)
            circle = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT,
                                      dp=dp,
                                      minDist=minDist,
                                      minRadius=minRadius,
                                      maxRadius=maxRadius,
                                      param1=param1,
                                      param2=param2)
            # print (circle)
            if circle is not None:
                circle_detected = True
                # print('circle detected with ', param1, param2)
                break
        if circle_detected == True:
            break

    if circle is not None:
        circle = np.round(circle[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        print('detected circle: ', circle)
        for (x, y, r) in circle:
            cv2.circle(output, (x, y), r, (255, 255, 255), 2)  # x,y,radius
            cv2.circle(output, (x, y), (r - distance_from_border), (255, 255, 255), 2)  # x,y,radius
            cv2.rectangle(output, (x - 2, y - 2), (x + 2, y + 2), (255, 255, 255), -1)

    # the area of the image with the largest intensity value
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(output)
    print(minVal, maxVal)

    plt.imshow(output, cmap='hot', vmin=minVal, vmax=maxVal)
    print('param1: ', param1)
    print('param2: ', param2)
    return param1, param2


def detect_timelapse_circles(detect_img, measure_img, dp, minDist, minRadius, maxRadius, param1, param2, distance_from_border = 5):
    circles_frame = []
    circle_frame_average = []
    circle_frame_stdev = []
    circle_frame_median = []
    circle_frame_lower = []
    circle_frame_upper = []
    for index, frame in enumerate(detect_img):
        image = frame[0]
        if image.dtype == 'uint16':
            print('image is uint16, converting to uint8 ...')
            image = img_as_ubyte(image)
            print('done converting to uint8')

        # print(index)
        output = image.copy()  # output on vis image
        # detect circles in the image
        # circle = [x,y,radius]
        circle = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT,  # detection on vis image
                                  dp=dp,
                                  minDist=minDist,
                                  minRadius=minRadius,
                                  maxRadius=maxRadius,  # todo: include min and max radius in input
                                  param1=param1,
                                  param2=param2)
        ic(circle)

        # tune conditions for circle detection to actually find circle
        if circle is None:
            for param1x in range(param1, 5, -2):
                print(param1x)
                circle = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT,  # detection on vis image
                                          dp=dp,
                                          minDist=minDist,
                                          minRadius=minRadius,
                                          maxRadius=maxRadius,
                                          param1=param1x,
                                          param2=param2)
                if param1x == 7:
                    print('no circle found')
                    break

                if circle is not None:
                    print('circle with new param1 found')
                    break

        # draw circle on frame
        if circle is not None:
            circle = np.round(circle[0, :]).astype("int")
            # loop over the (x, y) coordinates and radius of the circles
            #print('circles: ', circle)

            pixels_in_circle = []
            # loop through all circles and measure intensity inside
            for (x_coord, y_coord, radius) in circle:
                cv2.circle(output, (x_coord, y_coord), radius, (255, 255, 255), 2)  # x,y,radius
                cv2.rectangle(output, (x_coord - 2, y_coord - 2), (x_coord + 2, y_coord + 2), (255, 255, 255), -1)
                # ic(x,y,r)

                x_0 = x_coord
                y_0 = y_coord
                # make radius slighly smaller so border is not in range
                measurement_radius = radius - distance_from_border

                for x in range(x_0 - measurement_radius, x_0 + measurement_radius):
                    for y in range(y_0 - measurement_radius, y_0 + measurement_radius):
                        dx = x - x_coord
                        dy = y - y_coord
                        distanceSquared = dx ** 2 + dy ** 2
                        # print (distanceSquared)
                        if distanceSquared <= (measurement_radius ** 2):
                            # img_five[0][0] = first image, fluorescein channel
                            pixel_val = measure_img[index][0][dy][dx]  # measurement on GFP image (= 0)
                            pixels_in_circle.append(pixel_val)
        else:
            print('circle is none')

        #ic(pixels_in_circle)
        print('average: ', np.mean(pixels_in_circle))
        circle_frame_average.append(np.mean(pixels_in_circle))
        circle_frame_stdev.append(np.std(pixels_in_circle))

        circle_frame_median.append(np.median(pixels_in_circle))#, axis=0))
        circle_frame_lower.append(np.percentile(pixels_in_circle, 25, axis=0))
        circle_frame_upper.append(np.percentile(pixels_in_circle, 75, axis=0))

        circles_frame.append(circle)

        plt.figure(figsize=(10, 10))
        plt.imshow(output, cmap='hot')

    return circle_frame_average, circle_frame_stdev, circle_frame_median, circle_frame_lower, circle_frame_upper

if __name__ == '__main__':
    print('yay')
