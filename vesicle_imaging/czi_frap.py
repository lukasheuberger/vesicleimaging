import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import scipy.special
from icecream import ic
import pandas as pd
import h5py
import pickle

from vesicle_imaging import czi_image_handling as handler


def init_data(path, bleach_frame, bleach_channel, recovery_end_frame):
    """

    Args:
        path:
        bleach_frame:
        bleach_channel:
        recovery_end_frame:

    Returns:
        object:

    """
    os.chdir(path)
    # ic(os.listdir(path))
    ic(os.getcwd())

    # check if hdf5 file exists and load that instead of all data
    if os.path.exists('data.h5'):
        print('data file already exists, loading data ...')
        hf = h5py.File('data.h5', 'r')
        ic(hf.keys())
        image_cxyz_data = np.array(hf.get('image_cxyz_data'))
        deltaTs = np.array(hf.get('deltaTs'))
        scalings = np.array(hf.get('scalings'))
        filenames = np.array(hf.get('filenames'))
        hf.close()
        filenames = [filename.decode() for filename in filenames]

        frap_pickle = open("frap.pkl", "rb")
        frap_positions = pickle.load(frap_pickle)
        # ic(frap_positions)
        frap_pickle.close()

        metadata_pickle = open("metadata.pkl", "rb")
        image_metadata = pickle.load(metadata_pickle)
        metadata_pickle.close()

    else:
        files, filenames = handler.get_files(path)
        filenames.sort(reverse=True)
        files.sort(reverse=True)
        ic(files)

        print('number of images: ', len(files))

        # filenames = files
        # filenames.sort()
        # for i in range(0,len(files)):
        #     filenames[i] = files[i].replace('.czi','')
        # ic(filenames)

        image_data = []
        image_metadata = []
        image_add_metadata = []
        image_cxyz_data = []
        for file in files:
            ic(file)
            data, metadata, add_metadata = handler.load_image_data([file])
            image_data.append(data)
            image_metadata.append(metadata)
            image_add_metadata.append(add_metadata)
            img_data = handler.extract_channels_timelapse_xyt(data, bleach_channel)
            ic(img_data[0][bleach_frame:recovery_end_frame].shape) # make all same length so the can be exported to hdf5
            image_cxyz_data.append(img_data[0][0:recovery_end_frame])
        print('IMAGE IMPORT DONE!')

        # ic(image_data[0][0].shape)  # exemplary image shape
        # ic(len(image_cxyz_data))
        # #ic(image_cxyz_data.dtype)
        #
        # ic(image_cxyz_data[0][bleach_frame, recovery_end_frame])
        # ic(image_cxyz_data[0].shape)
        # ic(image_cxyz_data[0][0].shape)

        # handler.disp_basic_img_info(image_cxyz_data, image_metadata)  # exemplary basic metadata

        # to save images as pngs
        saving_on = True # todo: make that this actually does something
        if saving_on:
            try:
                os.mkdir('FRAP_analysis')  # todo make that this goes to the location of the input file
            except FileExistsError:
                pass

        frap_positions = []
        deltaTs = []
        scalings = []
        filenames = []

        # ic(image_cxyz_data)
        for index, img in enumerate(image_cxyz_data):
            # # save frame after bleaching
            # plt.figure()
            # plt.imshow(img[0][:,bleach_frame,:,:][bleach_channel])
            # savename = image_metadata[index][0]['Filename'].replace('.czi','_bleachframe.png')
            # savename = ''.join(['FRAP_analysis/', savename])
            # plt.savefig(savename, dpi=300)
            # plt.show()

            # read ROI info from metadata
            # ic(image_add_metadata[0][0]['Layers']['Layer'][0]['Elements']['Circle']['Geometry'])
            frap_position = dict(image_add_metadata[0][0]['Layers']['Layer'][0]['Elements']['Circle'][
                                     'Geometry'])  # centerx, centery, radius
            delta_time = \
                image_add_metadata[0][0]['DisplaySetting']['Information']['Image']['Dimensions']['Channels']['Channel'][0][
                    'LaserScanInfo']['FrameTime']

            scaling = float(handler.disp_scaling(image_add_metadata)[0])
            # ic(frap_position, delta_time, scaling)

            frap_positions.append(frap_position)
            deltaTs.append(delta_time)
            scalings.append(scaling)

            # save frame after bleaching with ROI
            plt.figure()
            ic(image_metadata[index][0]['Filename'])
            ic(img.shape)
            #output = img[0][bleach_frame, :, :].copy()
            output = img[bleach_frame, :, :].copy()
            cv2.circle(output, (int(float(frap_position['CenterX'])), int(float(frap_position['CenterY']))),
                       int(float(frap_position['Radius'])), (0, 0, 0), 2)  # x,y,radius
            plt.title(image_metadata[index][0]['Filename'].replace('.czi', ''))
            plt.imshow(output)
            filenames.append(image_metadata[index][0]['Filename'])
            savename_roi = image_metadata[index][0]['Filename'].replace('.czi', '_bleachframe_ROI.png')
            savename_roi = ''.join(['FRAP_analysis/', savename_roi])
            plt.savefig(savename_roi, dpi=300)
            # plt.show()
            plt.close()

        # ic(len(image_cxyz_data))
        # ic(len(image_cxyz_data[0]))
        # ic(image_cxyz_data[0][0].shape)
        # ic(image_cxyz_data[0][0].dtype)
        # ic(image_cxyz_data[0][0].astype(np.float64).dtype)

        hf = h5py.File('data.h5', 'w')
        hf.create_dataset('image_cxyz_data', data=image_cxyz_data)
        hf.create_dataset('deltaTs', data=deltaTs)
        hf.create_dataset('scalings', data=scalings)
        hf.create_dataset('filenames', data=filenames)
        hf.close()

        frap_pickle = open("frap.pkl", "wb")
        pickle.dump(frap_positions, frap_pickle)
        frap_pickle.close()

        metadata_pickle = open("metadata.pkl", "wb")
        pickle.dump(image_metadata, metadata_pickle)
        metadata_pickle.close()

        print('data saved to HDF5 and pickle files')

    return image_cxyz_data, frap_positions, deltaTs, scalings, image_metadata, filenames

    # ic(frap_position)
    # #ic(frap_position['CenterX'])
    #
    # frap_position = dict(frap_position)
    # ic(frap_position)
    #
    # ic(int(float(frap_position['CenterX'])))


# todo: make a dataframe with all filenames and params (frap positions, scaling, etc) -> export this as csv

def measure_fluorescence(image_cxyz_data, frap_positions, deltaTs, scalings, image_metadata, bleach_frame,
                         recovery_end_frame, norm = True):
    """

    Args:
        image_cxyz_data:
        frap_positions:
        deltaTs:
        scalings:
        image_metadata:
        bleach_frame:
        recovery_end_frame:
        norm:

    Returns:

    """
    fluorescence = []
    recovery = []
    bleach = []
    radii = []

#todo make full range (since range already cut)

    for index, img in enumerate(image_cxyz_data):
        #ic(img.shape)
        #ic(len(img))
        num_frames = len(img[:, :, :])
        ic(num_frames)
        stats = []

        x_0 = int(float(frap_positions[index]['CenterX']))
        y_0 = int(float(frap_positions[index]['CenterY']))
        radius_px = int(float(frap_positions[index]['Radius']))  # todo correct rounding
        # ic(x_0, y_0, radius_px)
        radius_m = float(frap_positions[index]['Radius']) * scalings[index]  # in m
        radius_um = radius_m * 10 ** 6  # convert to um
        # ic(radius_px, radius_m, radius_um)
        radii.append(radius_um)

        for frame in range(0, num_frames):
            measurement_image = img[frame, :, :]   # .copy()

            #measurement_image = img[0][frame, :, :]   # .copy()
            #ic(measurement_image.shape)

            pixels_in_circle = []
            #blank_image = np.zeros((264, 264, 3), np.uint8)

            for x in range(x_0 - radius_px, x_0 + radius_px):
                for y in range(y_0 - radius_px, y_0 + radius_px):
                    dx = x - x_0
                    dy = y - y_0

                    distance_squared = dx ** 2 + dy ** 2

                    if distance_squared <= (radius_px ** 2):
                        pixel_val = measurement_image[y][x]
                        pixels_in_circle.append(pixel_val)
                        #cv2.circle(blank_image, (x, y), radius=0, color=(255, 0, 255), thickness=-1)
            #plt.figure()
            #plt.imshow(blank_image)
            #plt.show()

            # print('filename: ', filenames[index])
            # print('no. of GUVs counted: ', len(detected_circles[index]))
            # print('number of pixels: ', len(pixels_in_circle))
            # print('min: ', np.min(pixels_in_circle))
            # print('max: ', np.max(pixels_in_circle))
            # print('average: ', np.mean(pixels_in_circle))
            # print('stdev: ', np.std(pixels_in_circle))
            stats.append(np.mean(pixels_in_circle))
        fluorescence.append(stats)

        plt.figure()
        plt.plot(stats)
        plt.show()

        ble = stats[0:bleach_frame]
        rec = stats[bleach_frame:num_frames]
        # ic(len(ble), len(rec))
        bleach_norm = ble / np.mean(ble)
        recovery_norm = rec / np.mean(ble)

        # default return normalized values
        if norm:
            recovery.append(recovery_norm)
            bleach.append(bleach_norm)
        else:
            recovery.append(stats[bleach_frame:num_frames])
            bleach.append(ble)
        # ic(recovery[0].shape, bleach[0].shape)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.tight_layout(pad=2)
        fig.suptitle(image_metadata[index][0]['Filename'].replace('.czi', ''))
        ax1.plot(ble, 'r-', label='bleach')
        ax1.plot(rec, 'b-', label='recovery')
        ax1.set(ylabel='fluorescence [a.u.]', xlabel='frame')
        ax2.plot(bleach_norm, 'r-', label='bleach')
        ax2.plot(recovery_norm, 'b-', label='recovery')
        ax2.set(ylabel='relative fluorescence [a.u.]', xlabel='frame')
        savename_fluo = image_metadata[index][0]['Filename'].replace('.czi', '_fluo_over_time.png')
        savename_fluo = ''.join(['FRAP_analysis/', savename_fluo])
        plt.legend()
        plt.savefig(savename_fluo, dpi=300)
        plt.close()
        # plt.show()

    # assume all experiments have same timing
    time_range = range(bleach_frame, num_frames)
    # ic(len(time_range))
    time = [timestep * float(deltaTs[0]) for timestep in time_range]  # convert to actual seconds

    return time, bleach, recovery, radii


def fit_data(time, recovery, radii, filenames):
    ic(len(recovery))
    # ic(recovery.shape)
    """

    Args:
        time:
        recovery:
        radii:

    Returns:

    """
    diffusion_constants = []

    def soumpasis_model(d0, t):  # , radius):
        tau = d0

        # F(t) = np.exp(-2*tau/t) * (scipy.special.i0(2*tau/t) + scipy.special.i1(2*tau/t))

        # ic(tau)
        # return np.exp(-2*tau/t) * (scipy.special.iv(0, 2*tau/t) + scipy.special.iv(1, 2*tau/t)) #traditional bessel
        # return np.exp(-2*((radius ** 2) / (4 * D))/t) * (scipy.special.i0(2*(radius ** 2 / (4 * D))/t)
        # + scipy.special.i1(2*(radius ** 2 / (4 * D))/t))

        return np.exp(-2 * tau / t) * (scipy.special.i0(2 * tau / t) + scipy.special.i1(2 * tau / t))  # fast bessel

    def residuals(d0, norm_intensity, t):
        """

        Args:
            d0:
            norm_intensity:
            t:

        Returns:

        """
        # ic(soumpasis_model(d0, t))
        return norm_intensity - soumpasis_model(d0, t)

    d0 = [10]  # initial guess

    for index, experiment in enumerate(recovery):
        # ic(experiment.shape, len(time))
        print('fitting: ', filenames[index])
        # area = radii[index]**2*np.pi #in um^2

        coeffs, cov = scipy.optimize.leastsq(residuals, d0, args=(experiment, time))

        diff_const = radii[index] ** 2 / (4 * coeffs)
        ic(diff_const)
        diffusion_constants.append(diff_const[0])

        plt.figure()
        plt.plot(time, experiment, 'k-')
        plt.plot(time, soumpasis_model(coeffs, time), 'r-')
        plt.xlabel('time (s)')
        plt.ylabel('norm. fluorescence (a.u.)')
        plt.ylim(top=1)
        # plt.saveig
        plt.show()

    return diffusion_constants


if __name__ == '__main__':
    # path = input('path to data folder: ')
    data_path = '/Users/lukasheuberger/local temp/analysis'
    # data_path = '/Users/lukasheuberger/code/phd/vesicle-imaging/test_data'
    bleach_frame = 5  # frame just after bleaching
    recovery_end_frame = 100
    bleach_channel = 0

    image_cxyz_data, frap_positions, deltaTs, scalings, metadata, filenames = init_data(data_path, bleach_frame,
                                                                                        bleach_channel, recovery_end_frame)
    ic(image_cxyz_data[0].shape)
    time, bleach, recovery, radii = measure_fluorescence(image_cxyz_data, frap_positions, deltaTs, scalings, metadata,
                                                         bleach_frame, recovery_end_frame)
    diffusion_constants = fit_data(time, recovery, radii, filenames)
    ic(diffusion_constants)

    df = pd.DataFrame({'filename' : filenames, 'diffusion constants' : diffusion_constants})
    df.to_excel('analysis.xlsx', index = False)

    # scatter_x = range(0, len(diffusion_constants))
    # plt.figure()
    # plt.scatter(scatter_x, diffusion_constants)
    # plt.show()
# todo: type hints