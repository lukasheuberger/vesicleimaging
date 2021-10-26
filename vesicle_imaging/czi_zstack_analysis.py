import matplotlib.pyplot as plt
import numpy as np
import cv2

def disp_zstack(image_stack: list, input_cmap: str):
    for index, image in enumerate(image_stack):
        plt.figure(figsize = (5,5))
        plt.axis('off')
        slice_title = ''.join(['slice ', str(index)])
        plt.title(slice_title)
        plt.imshow(image, cmap = input_cmap)

def detect_params(zstack, distance_from_border, zimage, dp, minDist, minRadius, maxRadius, param1top, param2top):

    # todo: include min and max radius in input from image data
    # distance_from_border = 50  # in px

    #param1_range = range(param1top, 5, -5)
    #param2_range = range(param2top, 150, 10)

    image = zstack[zimage]
    plt.figure(figsize=(10, 10))
    # plt.imshow(image, cmap = 'gray')
    output = image.copy()  # output on vis image
    # detect circles in the image
    
    circle_detected = False

    for param2 in range(param2top, 150, -10):
        for param1 in range(param1top, 5, -5):
            circle = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT,  # detection on vis image
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

    # if circle is not None:
    circle = np.round(circle[0, :]).astype("int")
    # loop over the (x, y) coordinates and radius of the circles
    print('detected circle: ', circle)
    for (x, y, r) in circle:
        cv2.circle(output, (x, y), r, (255, 255, 255), 2)  # x,y,radius
        cv2.circle(output, (x, y), (r - distance_from_border), (255, 255, 255), 2)  # x,y,radius
        cv2.rectangle(output, (x - 2, y - 2), (x + 2, y + 2), (255, 255, 255), -1)

    plt.imshow(output, cmap='hot')
    print ('param1: ', param1)
    print ('param2: ', param2)
    return param1, param2

def detect_stack_circles(zstack, dp, minDist, minRadius, maxRadius, param1, param2):
    circles_zstack = []
    for index, image in enumerate(zstack):
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
                print (param1x)
                circle = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT,  # detection on vis image
                                          dp=dp,
                                          minDist=minDist,
                                          minRadius=minRadius,
                                          maxRadius=maxRadius,
                                          param1=param1x,
                                          param2=param2)
                if param1x == 7:
                    print ('no circle found')
                    break
                    
                if circle is not None:
                    print ('circle with new param1 found')
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

def measure_intensity_circle(circles, detection_image, distance_from_border ):
    # intensity in circles
    average_per_img = []

    for index, img in enumerate(detection_image):
        # plt.imshow(image[0])
        output = img.copy()  # output on vis image
        pixels_in_circle = []
        print('slice: ', index)

        if circles[index] is not None:
            for circle in circles[index]:

                print(circle)
                # test_circle = [ 163, 1487,  108] # x,y,radius
                x_0 = circle[0]
                y_0 = circle[1]
                radius = circle[2]

                # make radius slighly smaller so border is not in range
                measurement_radius = radius - distance_from_border
                # print (measurement_radius)

                for x in range(x_0 - measurement_radius, x_0 + measurement_radius):
                    for y in range(y_0 - measurement_radius, y_0 + measurement_radius):
                        dx = x - circle[0]
                        dy = y - circle[1]
                        distanceSquared = dx ** 2 + dy ** 2
                        # print (distanceSquared)
                        if distanceSquared <= (measurement_radius ** 2):
                            # img_five[0][0] = first image, fluorescein channel
                            pixel_val = img[dy][dx]  # measurement on GFP image (= 0)
                            pixels_in_circle.append(pixel_val)
                cv2.circle(output, (x_0, y_0), (measurement_radius), (255, 255, 255), 2)  # x,y,radius
            print('slice: ', index)
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

    #    fig = plt.figure(figsize=(10, 10),frameon=False)
    #    fig.tight_layout(pad=0)
    #    plt.imshow(output, cmap = 'hot')
    #    plt.axis('off')
    # plt.imsave('output.png',output,cmap='hot')

    return average_per_img