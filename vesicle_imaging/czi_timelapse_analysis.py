import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage import img_as_ubyte


def detect_params(timelapse, distance_from_border = 5, timepoint = 5, dp = 2, minDist = 10, minRadius = 5, maxRadius = 50, param1top = 50, param2top = 300):
    # todo: include min and max radius in input from image data
    # distance_from_border = 50  # in px
    # param1_range = range(param1top, 5, -5)
    # param2_range = range(param2top, 150, 10)

    image = timelapse[timepoint][0]
    plt.figure(figsize=(5, 5))
    plt.axis('off')
    plt.imshow(image, cmap = 'gray')
    output = image.copy()  # output on vis image
    # detect circles in the image

    circle_detected = False

    if image.dtype == 'uint16':
        print('image is uint16, converting to uint8 ...')
        image = img_as_ubyte(image)
        print('done converting to uint8')

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

    plt.imshow(output, cmap='hot')
    print('param1: ', param1)
    print('param2: ', param2)
    return param1, param2


def detect_timelapse_circles(timelapse, dp, minDist, minRadius, maxRadius, param1, param2):
    circles_zstack = []
    for index, image in enumerate(timelapse):
        image = image[0]
        if image.dtype == 'uint16':
            print('image is uint16, converting to uint8 ...')
            image = img_as_ubyte(image)
            print('done converting to uint8')

        print(index)
        output = image.copy()  # output on vis image
        # detect circles in the image
        circle = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT,  # detection on vis image
                                  dp=dp,
                                  minDist=minDist,
                                  minRadius=minRadius,
                                  maxRadius=maxRadius,  # todo: include min and max radius in input
                                  param1=param1,
                                  param2=param2)
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

        if circle is not None:
            circle = np.round(circle[0, :]).astype("int")
            # loop over the (x, y) coordinates and radius of the circles
            print('circle: ', circle)
            for (x, y, r) in circle:
                cv2.circle(output, (x, y), r, (255, 255, 255), 2)  # x,y,radius
                cv2.rectangle(output, (x - 2, y - 2), (x + 2, y + 2), (255, 255, 255), -1)
        else:
            print('circle is none')

        circles_zstack.append(circle)

        plt.figure(figsize=(10, 10))
        plt.imshow(output, cmap='hot')

    return circles_zstack