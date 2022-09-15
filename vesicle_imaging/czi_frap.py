import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import scipy.special
from icecream import ic

from vesicle_imaging import czi_image_handling as handler


def init_data(path, bleach_frame, bleach_channel):

    os.chdir(path)
    #ic(os.listdir(path))
    ic(os.getcwd())

    files, filenames = handler.get_files(path)
    filenames.sort(reverse=True)
    files.sort(reverse=True)
    ic(files)

    #print (images)
    print ('number of images: ', len(files))

    # filenames = files
    # filenames.sort()
    # for i in range(0,len(files)):
    #     filenames[i] = files[i].replace('.czi','')
    # ic(filenames)


    image_data = []
    image_metadata = []
    image_add_metadata = []
    image_cxyz_data = []
    for image in files:
        ic(image)
        data, metadata, add_metadata = handler.load_image_data([image])
        image_data.append(data)
        image_metadata.append(metadata)
        image_add_metadata.append(add_metadata)

        img_data = handler.extract_channels_timelapse(data)
        image_cxyz_data.append(img_data)
    print ('IMAGE IMPORT DONE!')

    ic(image_data[0][0].shape) # exemplary image shape
    ic(image_cxyz_data[0][0].shape)

    handler.disp_basic_img_info(image_cxyz_data[0], image_metadata) # exemplary basic metadata

    # to save images as pngs
    saving_on = True
    if saving_on == True:
        try:
            os.mkdir('FRAP_analysis')#todo make that this goes to the location of the input file
        except FileExistsError:
            pass

    # channels = ['rhoPE', 'Transmission']
    #ic(image_cxyz_data[0][:,0,:,:].shape)

    frap_positions = []
    deltaTs = []
    scalings = []

    #ic(image_cxyz_data)
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
        deltaT = \
        image_add_metadata[0][0]['DisplaySetting']['Information']['Image']['Dimensions']['Channels']['Channel'][0][
            'LaserScanInfo']['FrameTime']

        scaling = float(handler.disp_scaling(image_add_metadata)[0])
        #ic(frap_position, deltaT, scaling)

        frap_positions.append(frap_position)
        deltaTs.append(deltaT)
        scalings.append(scaling)

        # save frame after bleaching with ROI
        plt.figure()
        output = img[0][:,bleach_frame,:,:][bleach_channel].copy()
        cv2.circle(output, (int(float(frap_position['CenterX'])), int(float(frap_position['CenterY']))),
                   int(float(frap_position['Radius'])), (0, 0, 0), 2)  # x,y,radius
        plt.title(image_metadata[index][0]['Filename'].replace('.czi',''))
        plt.imshow(output)
        savename_ROI = image_metadata[index][0]['Filename'].replace('.czi','_bleachframe_ROI.png')
        savename_ROI = ''.join(['FRAP_analysis/', savename_ROI])
        plt.savefig(savename_ROI, dpi=300)
        # plt.show()
        plt.close()

    return image_cxyz_data, frap_positions, deltaTs, scalings, image_metadata

        # ic(frap_position)
        # #ic(frap_position['CenterX'])
        #
        # frap_position = dict(frap_position)
        # ic(frap_position)
        #
        # ic(int(float(frap_position['CenterX'])))

#todo: make a dataframe with all filenames and params (frap positions, scaling, etc) -> export this as csv

def measure_fluorescence(image_cxyz_data, frap_positions, deltaTs, scalings, image_metadata, bleach_frame, recovery_end_frame, norm=True):
    fluorescence = []
    recovery = []
    bleach = []
    radii = []

    for index, img in enumerate(image_cxyz_data):
        num_frames = len(img[0][:,:,:,:][bleach_channel])

        stats = []

        x_0 = int(float(frap_positions[index]['CenterX']))
        y_0 = int(float(frap_positions[index]['CenterY']))
        radius_px = int(float(frap_positions[index]['Radius'])) #todo correct rounding
        # ic(x_0, y_0, radius_px)
        radius_m = float(frap_positions[index]['Radius'])*scalings[index] # in m
        radius_um = radius_m * 10**6 # convert to um
        # ic(radius_px, radius_m, radius_um)
        radii.append(radius_um)

        for frame in range(0, num_frames):
            #ic(frame)
            measurement_image = img[0][:,frame,:,:][0]#.copy()

            pixels_in_circle = []

            for x in range(x_0 - radius_px, x_0 + radius_px):
                for y in range(y_0 - radius_px, y_0 + radius_px):
                    dx = x - x_0
                    dy = y - y_0

                    distanceSquared = dx ** 2 + dy ** 2

                    if distanceSquared <= (radius_px ** 2):
                        pixel_val = measurement_image[y][x]
                        pixels_in_circle.append(pixel_val)
                        #cv2.circle(blank_image, (x, y), radius=0, color=(255, 0, 255), thickness=-1)
            # plt.figure()
            # plt.imshow(blank_image)
            # plt.show()

            #print('filename: ', filenames[index])
            #print('no. of GUVs counted: ', len(detected_circles[index]))
            #print('number of pixels: ', len(pixels_in_circle))
            # print('min: ', np.min(pixels_in_circle))
            # print('max: ', np.max(pixels_in_circle))
            # print('average: ', np.mean(pixels_in_circle))
            # print('stdev: ', np.std(pixels_in_circle))
            stats.append(np.mean(pixels_in_circle))
        fluorescence.append(stats)

        ble = stats[0:bleach_frame]
        rec = stats[bleach_frame:recovery_end_frame]
        bleach_norm = ble / np.mean(ble)
        recovery_norm = rec / np.mean(ble)


        # default return normalized values
        if norm:
            recovery.append(recovery_norm)
            bleach.append(bleach_norm)
        else:
            recovery.append(stats[bleach_frame:recovery_end_frame])
            bleach.append(ble)

        fig, (ax1, ax2) = plt.subplots(1,2)
        fig.tight_layout(pad=2)
        fig.suptitle(image_metadata[index][0]['Filename'].replace('.czi',''))
        ax1.plot(ble,'r-', label = 'bleach')
        ax1.plot(rec,'b-', label = 'recovery')
        ax1.set(ylabel='fluorescence [a.u.]', xlabel='frame')
        ax2.plot(bleach_norm,'r-', label = 'bleach')
        ax2.plot(recovery_norm,'b-', label = 'recovery')
        ax2.set(ylabel='relative fluorescence [a.u.]', xlabel='frame')
        savename_fluo = image_metadata[index][0]['Filename'].replace('.czi','_fluo_over_time.png')
        savename_fluo = ''.join(['FRAP_analysis/', savename_fluo])
        plt.legend()
        plt.savefig(savename_fluo, dpi=300)
        plt.close()
        # plt.show()

    # assume all experiments have same timing
    time_range = range(bleach_frame, recovery_end_frame)
    time = [timestep * float(deltaTs[index]) for timestep in time_range] # convert to actual seconds

    return time, bleach, recovery, radii

def fit_data(time, recovery, radii):

    diffusion_constants = []

    def Soumpasis_model(d0, t):#, radius):
        tau = d0
        #ic(tau)
        #F(t) = np.exp(-2*tau/t) * (scipy.special.i0(2*tau/t) + scipy.special.i1(2*tau/t))
        #return np.exp(-2*tau/t) * (scipy.special.iv(0, 2*tau/t) + scipy.special.iv(1, 2*tau/t)) #traditional bessel
        #return np.exp(-2*((radius ** 2) / (4 * D))/t) * (scipy.special.i0(2*(radius ** 2 / (4 * D))/t) + scipy.special.i1(2*(radius ** 2 / (4 * D))/t))
        return np.exp(-2*tau/t) * (scipy.special.i0(2*tau/t) + scipy.special.i1(2*tau/t)) #fast bessel

    def residuals(d0, I_norm, t):
        return I_norm - Soumpasis_model(d0, t)

    d0 = [10] # initial guess

    for index, experiment in enumerate(recovery):

        # area = radii[index]**2*np.pi #in um^2
        coeffs, cov = scipy.optimize.leastsq(residuals, d0, args=(experiment, time))
        ic(cov)

        diff_const = radii[index] ** 2 / (4 * coeffs)
        ic(diff_const)
        diffusion_constants.append(diff_const[0])

        plt.figure()
        plt.plot(time, experiment, 'k-')
        plt.plot(time, Soumpasis_model(coeffs, time), 'r-')
        plt.xlabel('time (s)')
        plt.ylabel('norm. fluor. (a.u.)')
        plt.ylim(top = 1)
        plt.show()

    return diffusion_constants
    # # calculate area of circle
    # # ic(float(frap_position['Radius']))
    # # area = radius_um**2*np.pi #in um^2
    # d_x = np.sqrt(area)
    # d_y = np.sqrt(area)
    # ic(area, d_x, d_y)
    #
    # # Define log posterior
    # def log_posterior(p, I_norm, t, d_x, d_y):
    #     """
    #     Return log of the posterior.
    #     """
    #     return -len(I_norm) / 2.0 * np.log(
    #             ((I_norm - norm_fluor_recov(p, t, d_x, d_y))**2).sum())
    #
    # # Define fit function
    # def norm_fluor_recov(p, time, d_x, d_y):
    #     """
    #     Return normalized fluorescence as function of time.
    #     """
    #     # Unpack parameters
    #     #f_b, f_f, D, k_off = p
    #     f_f, f_b, D, k_off = p
    #
    #     # Function to compute psi
    #     def psi(time_array, D, d_i):
    #         return [d_i / 2.0 * scipy.special.erf(d_i / np.sqrt(4.0 * D * t)) \
    #                      - np.sqrt(D * t / np.pi) \
    #                                 * (1.0- np.exp(-d_i**2 / (4.0 * D * t))) for t in time_array]
    #     psi_x = psi(time, D, d_x)
    #     psi_y = psi(time, D, d_y)
    #     return f_f * (1.0 - f_b * 4.0 * np.exp([-k_off * t for t in time]) / d_x / d_y * psi_x * psi_y)
    #     #todo: make consistent with naming t vs t_array
    #
    # # Define residual
    # def resid(p, I_norm, t, d_x, d_y):
    #     return I_norm - norm_fluor_recov(p,t, d_x, d_y)
    #
    # # Perform the curve fit
    # p0 = np.array([0.9, 0.9, 10.0, 0.1])
    # #p0 = np.array([0.9, 0.9, 10.0, 0.1])
    # popt, pcov = scipy.optimize.leastsq(resid, p0, args=(recovery1_norm, t1, d_x, d_y))
    # #popt, junk_output = scipy.optimize.leastsq(resid, p0, args=(I_norm, t, d_x, d_y))
    # ic(popt)
    # s_sq = (resid(popt, recovery1_norm, t1, d_x, d_y)**2).sum()/(len(recovery1_norm)-len(p0))
    # pcov = pcov * s_sq
    # ic(s_sq, pcov)


    # errfunc = lambda p, x, y: norm_fluor_recov(x,p) - y
    # s_sq = (errfunc(popt, t1, recovery1_norm)**2).sum()/(len(recovery1_norm)-len(p0))
    # pcov = pcov * s_sq
    # perr = np.sqrt(np.diag(pcov))
    # ic(pcov, perr)

    # Compute the covariance
    #hes = jb.hess_nd(log_posterior, popt, args=(I_norm, t, d_x, d_y))
    #cov = -np.linalg.inv(hes)

    # Report results
    #     formats = tuple(popt) #+ tuple(np.sqrt(np.diag(pcov)))
    # # print("""
    # # f_f = {0:.3f} +- {4:.3f}
    # # f_b = {1:.3f} +- {5:.3f}
    # # D = {2:.3f} +- {6:.3f} µm^2/s
    # # k_off = {3:.3f} +- {7:.3f} (1/s)
    # # """.format(*formats))
    # print("""
    # f_f = {0:.3f}
    # f_b = {1:.3f}
    # D = {2:.3f} µm^2/s
    # k_off = {3:.3f} (1/s)
    # """.format(*formats))
    # #todo: make that this works for covariance
    # #todo: also, D might be wrong
    #
    # # Plot recovery trace
    # plt.plot(t1, recovery1_norm, 'k-')
    # plt.plot(t1, norm_fluor_recov(popt, t1, d_x, d_y), 'r-')
    # plt.xlabel('time (s)')
    # plt.ylabel('norm. fluor. (a.u.)')
    # plt.show()
    #

if __name__ == '__main__':
    #path = input('path to data folder: ')
    path = '/Users/heuberger/code/vesicle-imaging/test_data'
    bleach_frame = 5 #frame just after bleaching
    recovery_end_frame = 100
    bleach_channel = 0

    image_cxyz_data, frap_positions, deltaTs, scalings, metadata = init_data(path, bleach_frame, bleach_channel)
    time, bleach, recovery, radii = measure_fluorescence(image_cxyz_data, frap_positions, deltaTs, scalings, metadata, bleach_frame, recovery_end_frame)
    ic(recovery)

    diffusion_constants = fit_data(time, recovery, radii)
    ic(diffusion_constants)

    # scatter_x = range(0, len(diffusion_constants))
    # plt.figure()
    # plt.scatter(scatter_x, diffusion_constants)
    # plt.show()