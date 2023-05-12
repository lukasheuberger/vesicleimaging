"""Toolset to analyze FRAP experiments from czi files

This module contains functions to load and analyze FRAP
experiments from czi files. It loads all czi files and extracts
the metadata and image data. It then uses the metadata to extract
the positions of the FRAP circles and the scaling factors
and fits the FRAP curves to the corresponding diffusion models.

"""

import os
import pickle

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.special
from .file_handling import get_files
from .czi_processing import load_image_data, extract_channels
from .image_info import disp_scaling


def load_data(path: str):
    """
    The load_data function loads the data from a hdf5 file.
    It returns the image_cxyz_data, deltaTs and scalings as numpy arrays.

    Args:
        path:str: Specify the path to the hdf5 file

    Returns:
        A tuple of four elements
    """

    hf_file = h5py.File(path, "r")
    # ic(hf.keys())
    data = np.array(hf_file.get("image_cxyz_data"))
    time = np.array(hf_file.get("deltaTs"))
    scale = np.array(hf_file.get("scalings"))
    filenames = np.array(hf_file.get("filenames"))
    hf_file.close()

    filename = [filename.decode() for filename in filenames]

    with open("frap.pkl", "rb") as frap_pickle:
        frap = pickle.load(frap_pickle)

    with open("metadata.pkl", "rb") as metadata_pickle:
        metadata = pickle.load(metadata_pickle)

    return data, time, scale, filename, frap, metadata


def save_files(
        data: list[int],
        time: list[float],
        scale: list[float],
        filename: list[str],
        frap: list[float],
        metadata: list[str],
):
    """
    The save_files function saves the data from the image_cxyz_data,
    deltaTs, scalings and filenames to a hdf5 file called 'data.h5'.
    It also saves frap positions to a pickle file called 'frap.pkl'
    and metadata to a pickle file called 'metadata.pkl'.

    Args:
        data:list[int]: image data
        time:list[float]: time axis of data (in seconds)
        scale:list[float]: Scale the data
        filename:list[str]: filenames of image data files
        frap:list[float]: Save the frap_positions to a pickle file
        metadata:list[str]: Store the metadata of the experiment

    Returns:
        A tuple containing the data, time, scale,
        filename and frap variables
    """

    hf_file = h5py.File("data.h5", "w")
    hf_file.create_dataset("image_cxyz_data", data=data)
    hf_file.create_dataset("deltaTs", data=time)
    hf_file.create_dataset("scalings", data=scale)
    hf_file.create_dataset("filenames", data=filename)
    hf_file.close()

    # save frap_positions to pickle file
    with open("frap.pkl", "wb") as frap_pickle:
        pickle.dump(frap, frap_pickle)

    # save metadata to pickle file
    with open("metadata.pkl", "wb") as metadata_pickle:
        pickle.dump(metadata, metadata_pickle)


def init_data(path: str, bleach: int, channel_bleach: int, recovery_frame: int):
    """
    The init_data function imports the image data and metadata from a
    folder containing .czi files. It extracts the FRAP ROI position,
    time delta between frames, and scaling factor for each image.
    The function returns all of these variables as outputs.

    Args:
        path:str: Define the path to the image files
        bleach:int: Select the frame after bleaching
        channel_bleach:int: Select the channel that is used for bleaching
        recovery_frame:int: number of frames to be analyzed after bleaching

    Returns:
        tbd
    """

    # Load the data
    os.chdir(path)
    # ic(os.listdir(path))
    print(f'current working directory: {os.getcwd()}')

    # check if hdf5 file exists and load that instead of all data
    if os.path.exists("data.h5"):
        print("data file already exists, loading data ...")
        (
            image_txy_data,
            delta_ts,
            scalings,
            filenames,
            frap_positions,
            image_metadata,
        ) = load_data("data.h5")

    else:
        files, _ = get_files(path)
        # ic(files)

        print("number of images: ", len(files))

        # init empty arrays for data storage
        image_data = []
        image_metadata = []
        image_add_metadata = []
        image_txy_data = []

        # iterate over all files and read files using handler package
        # extract only one channel (bleach_channel) and x,y data over time
        for file in files:
            # ic(file)
            data, metadata, add_metadata = load_image_data([file])
            image_data.append(data)
            image_metadata.append(metadata)
            image_add_metadata.append(add_metadata)
            img_data = extract_channels(data)[0]

            # cut all to time of interest for hdf5 export
            # ic(img_data[0][bleach:recovery_frame].shape)
            image_txy_data.append(img_data[0][0:recovery_frame])
        print("IMAGE IMPORT DONE!")

        # handler.disp_basic_img_info(data, metadata)

        # check if folder exists
        # if not: create folder to save images
        try:
            os.mkdir("FRAP_analysis")
        except FileExistsError:
            pass

        # initialize empty arrays for data storage
        frap_positions = []
        delta_ts = []
        scalings = []
        filenames = []

        for index, img in enumerate(image_data):
            # # save frame after bleaching
            # plt.figure()
            # plt.imshow(img[0][:,bleach_frame,:,:][bleach_channel])
            # savename = image_metadata[index][0]['Filename']
            #   .replace('.czi','_bleachframe.png')
            # savename = ''.join(['FRAP_analysis/', savename])
            # plt.savefig(savename, dpi=300)
            # plt.show()

            frap_position = dict(
                image_add_metadata[0][0]["Layers"]["Layer"][0]["Elements"]["Circle"][
                    "Geometry"
                ]
            )
            # frap_position = centerx, centery, radius

            # read ROI info from metadata
            frap_positions.append(frap_position)
            delta_ts.append(
                image_add_metadata[0][0]["DisplaySetting"]["Information"]["Image"][
                    "Dimensions"
                ]["Channels"]["Channel"][0]
                ["LaserScanInfo"]["FrameTime"]
            )
            scalings.append(float(disp_scaling(image_add_metadata)[0]))

            # ic(image_metadata[index][0]['Filename'])

            # save frame after bleaching with ROI
            plt.figure()
            output = img[0][0, 0, 0, bleach, 0, :, :].copy()

            cv2.circle(
                output,
                (
                    int(float(frap_position["CenterX"])),
                    int(float(frap_position["CenterY"])),
                ),
                int(float(frap_position["Radius"])),
                (0, 0, 0),
                2,
            )  # x,y,radius
            plt.title(image_metadata[index][0]["Filename"].replace(".czi", ""))
            plt.imshow(output)
            filenames.append(image_metadata[index][0]["Filename"])
            savename_roi = "".join(
                [
                    "FRAP_analysis/",
                    image_metadata[index][0]["Filename"].replace(
                        ".czi", "_bleachframe_ROI.png"
                    ),
                ]
            )
            plt.savefig(savename_roi, dpi=300)
            # plt.show()
            plt.close()

        # save image data to hdf5 file
        save_files(
            image_txy_data,
            delta_ts,
            scalings,
            filenames,
            frap_positions,
            image_metadata,
        )

        print("data saved to HDF5 and pickle files")

    return image_txy_data, frap_positions, delta_ts, scalings, image_metadata, filenames


def measure_fluorescence(
        image_data: list[int],
        frap_positions: list[dict],
        delta_t: list[float],
        scalings: list[float],
        image_metadata: list[dict],
        frame_bleach: int,
        normalize: bool = True,
):
    """
    The measure_fluorescence function takes a list of image data,
    a list of frap positions, a list of deltaTs (time between frames),
    and a scaling factor. It returns the time range (in s),
    bleaching curve and recovery curves for each experiment.

    Args:
        image_data:list: list of image data (returned by load_image function)
        frap_positions:list: list of frap positions per experiment
        delta_t:list: Convert the frame numbers to seconds
        scalings:list: scaling factor (px -> m)
        image_metadata:list: metadata of all images
        frame_bleach:int: frame at which bleaching has started
        normalize:bool: Normalize the fluorescence values to 1

    Returns:
        :
    """

    # initialize empty arrays for data storage
    fluorescence = []
    recovery = []
    bleach = []
    radii = []

    # iterate over all experiments in folder
    for index, experiment in enumerate(image_data):
        num_frames = len(experiment[:, :, :])
        experiment_statistics = []

        x_0 = int(float(frap_positions[index]["CenterX"]))
        y_0 = int(float(frap_positions[index]["CenterY"]))
        radius_px = int(round(float(frap_positions[index]["Radius"])))
        # ic(x_0, y_0, radius_px)

        # convert radius to other units
        radius_m = float(frap_positions[index]["Radius"]) * scalings[index]  # in m
        radius_um = radius_m * 10 ** 6  # convert to um
        # ic(radius_px, radius_m, radius_um)
        radii.append(radius_um)  # store as um

        for frame in range(0, num_frames):  # iterate over frames in experiment
            measurement_image = experiment[frame, :, :]
            pixels_in_circle = []

            # iterate over all pixels in bleached circle
            for x_coord in range(x_0 - radius_px, x_0 + radius_px):
                for y_coord in range(y_0 - radius_px, y_0 + radius_px):
                    delta_x = x_coord - x_0
                    delta_y = y_coord - y_0

                    distance_squared = delta_x ** 2 + delta_y ** 2
                    if distance_squared <= (radius_px ** 2):
                        pixel_val = measurement_image[0][y_coord][x_coord]
                        pixels_in_circle.append(pixel_val)

            # print('filename: ', filenames[index])
            # print('number of pixels: ', len(pixels_in_circle))
            # print('min: ', np.min(pixels_in_circle))
            # print('max: ', np.max(pixels_in_circle))
            # print('average: ', np.mean(pixels_in_circle))
            # print('stdev: ', np.std(pixels_in_circle))

            # append mean of all pixels
            experiment_statistics.append(np.mean(pixels_in_circle))
        fluorescence.append(experiment_statistics)

        fluorescence_baseline = experiment_statistics[0:frame_bleach]
        recovery_raw = experiment_statistics[frame_bleach:num_frames]

        # normalize bleaching to 1
        bleach_norm = fluorescence_baseline / np.mean(fluorescence_baseline)
        recovery_norm = recovery_raw / np.mean(fluorescence_baseline)

        # default return normalized values
        if normalize:
            recovery.append(recovery_norm)
            bleach.append(bleach_norm)
        else:
            recovery.append(recovery_raw)
            bleach.append(fluorescence_baseline)

        # plot raw and normalized data
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.tight_layout(pad=2)
        fig.suptitle(image_metadata[index][0]["Filename"].replace(".czi", ""))
        ax1.plot(fluorescence_baseline, "r-", label="bleach")
        ax1.plot(recovery_raw, "b-", label="recovery")
        ax1.set(ylabel="fluorescence [a.u.]", xlabel="frame")
        ax2.plot(bleach_norm, "r-", label="bleach")
        ax2.plot(recovery_norm, "b-", label="recovery")
        ax2.set(ylabel="relative fluorescence [a.u.]", xlabel="frame")
        savename_fluo = "".join(
            [
                "FRAP_analysis/",
                image_metadata[index][0]["Filename"].replace(
                    ".czi", "_fluo_over_time.png"
                ),
            ]
        )
        plt.legend()
        plt.savefig(savename_fluo, dpi=300)
        plt.close()
        # plt.show()

    # assume all experiments have same timing
    time_range = range(frame_bleach, len(image_data[0][:, :, :]))
    # ic(len(time_range))

    # convert to actual seconds
    time = [timestep * float(delta_t[0]) for timestep in time_range]

    return time, recovery, radii


def fit_data(
        time: list[float],
        recovery: list[float],
        radii: list[float],
        filenames: list[str],
        laser_profile: str,
):
    """
    The fit_data function takes a list of time values,
    a list of recovery values, a list of filenames and
    type of the laser profile.  It then fits these data to
    the appropriate model (uniform or gaussian) and returns
    an array containing the diffusion constants for each image.


    Args:
        time:list[float]: Calculate the model for a given time value
        recovery:list[float]: Pass the data from each image
        radii:list[float]: Define the size of the sample area
        filenames:list[str]: Pass a list of filenames to be analyzed
        laser_profile:str: Define the laser profile

    Returns:
        The fitted values of the diffusion coefficient, d0
    """

    def chi_squared(obs_freq, exp_freq):
        count = len(obs_freq)
        chi_sq = 0
        for i in range(count):
            x = (obs_freq[i] - exp_freq[i]) ** 2
            x = x / exp_freq[i]
            chi_sq += x
        return chi_sq

    def soumpasis_model(d_0: int, time: list[float], bessel_type: str = "fast"):
        """
        The soumpasis_model function takes a single parameter, d0,
        which is the initial value of the model.  It then takes a
        list of time values and returns an array of modeled values
        for each time value in the list.  The soumpasis_model function
        uses scipy's bessel functions to calculate
        F(t) = exp(-2*tau/t)*(I0(2*tau/t)+I0(2*d0/d))
        based on Soumpasis DM. Theoretical analysis of fluorescence
        photobleaching recovery experiments.
        Biophys J. 1983 Jan;41(1):95-7.
        doi: 10.1016/S0006-3495(83)84410-5.

        Args:
            d_0:int: initial value of tau
            time:list: list of time values

        Returns:
            The values of the model for a given value of
            d0 and a list of time values
        """

        # F(t) = np.exp(-2*tau/t) * (scipy.special.i0(2*tau/t)
        #                           + scipy.special.i1(2*tau/t))
        tau = np.array(d_0)

        if bessel_type == "fast":
            return np.exp(-2 * tau / time) * (
                    scipy.special.i0(2 * tau / time) + scipy.special.i1(2 * tau / time)
            )
        if bessel_type == "traditional":
            return np.exp(-2 * tau / time) * (
                    scipy.special.iv(0, 2 * tau / time)
                    + scipy.special.iv(1, 2 * tau / time)
            )
        raise ValueError('bessel_type must be either "fast" or "traditional"')

    def confocal_frap_model(
            k: float,
            d_0: int,
            time_array: list[float],
            radius: float,
            mobile_fract: int,
            f_0: float,
    ):
        """
        The confocal_frap_model function computes the FRAP curve
        for a given set of parameters.
        F(t) = (1-(k/(2+(8*d0*t)/(r**2)))) *
        mobile_fraction+(1-mobile_fraction)*F0

        Based on Kang M, Day CA, Kenworthy AK, DiBenedetto E.:
        Simplified equation to extract diffusion coefficients
        from confocal FRAP data.
        Traffic. 2012 Dec;13(12):1589-600. doi: 10.1111/tra.12008

        Args:
            k:float: fraction of molecules that are fluorescent at any time
            d_0:int: diffusion coefficient
            time_array:list: array of time
            radius:float: radius of bleached area (circle)
            mobile_fract:float: mobile fraction
            (mf0 = 1 means all fluorophores are in the ground state)
            f_0:float: Set the initial fluorescence intensity

        Returns:
            The expected frap intensity for a given radius, mf0 and f0

        """
        # ic(F0, k, d0, t, mobile_fraction,r)
        # ic([(1-(k/(2+(8*d0*t)/(r**2))))*
        #   mobile_fraction+(1-mobile_fraction)*F0 for t in time])
        return [
            (1 - (k / (2 + (8 * d_0 * timepoint) / (radius ** 2)))) * mobile_fract
            + (1 - mobile_fract) * f_0
            for timepoint in time_array
        ]

    def residuals(
            guess: list[int],
            norm_intensity: list[float],
            time_array: list[float],
            radius: float,
            init_fluo: float,
            k: float,
    ):
        """
        The residuals function computes the difference between the measured
        fluorescence intensity and the predicted fluorescence intensity from
        a model of the confocal FRAP experiment. The residuals are returned
        as an array, which is then used by scipy.optimize to minimize the sum
        of squared residuals.

        Args:
            guess:list[int]: Pass the initial guess for the parameters
            norm_intensity:list[float]: Normalize the data
            time_array:list[float]: array of time points
            radius:float: Define the radius of the laser beam
            init_fluo:float: Define the initial fluorescence intensity
            k:float: bleach depth parameter

        Returns:
            The difference between the measured and fitted data
        """

        [diff_0, mf0] = guess

        if laser_profile == "uniform":
            return norm_intensity - soumpasis_model(diff_0, time_array)
        if laser_profile == "gaussian":
            return norm_intensity - confocal_frap_model(
                k, diff_0, time_array, radius, mf0, init_fluo
            )
        raise ValueError(f"Unsupported laser profile: {laser_profile}")

    diffusion_constants = []

    # initial guess; diffusion coeff, mobile fraction
    params_guess = np.array([5, 10])

    # iterate over experiments
    for index, experiment in enumerate(recovery):
        print("fitting: ", filenames[index])

        # init_fluorescence = fluorescence before bleaching
        init_fluorescence = experiment[0]
        bleach_depth = 2 - 2 * init_fluorescence

        coeffs, _ = scipy.optimize.leastsq(
            residuals,
            params_guess,
            args=(experiment, time, radii[index], init_fluorescence, bleach_depth),
        )

        plt.figure()

        if laser_profile == "uniform":
            diff_const = radii[index] ** 2 / (4 * coeffs[0])
            plt.plot(time, soumpasis_model(coeffs[0], time), "r-", label="fit")

            # ic(chi_squared(experiment, soumpasis_model(coeffs[0], time)))
            # #ic(stats.chisquare(experiment, soumpasis_model(coeffs[0], time)))
            #
            # # critical Chi-Square - percent point function
            # p = 1
            # DOF = len(experiment) - p - 1
            # ic(stats.chi2.ppf(0.95, DOF))

        if laser_profile == "gaussian":
            diff_const = coeffs[0]
            plt.plot(
                time,
                confocal_frap_model(
                    bleach_depth,
                    coeffs[0],
                    time,
                    radii[index],
                    coeffs[1],
                    init_fluorescence,
                ),
                "r-",
                label="fit",
            )
            # todo: number seems off, check again

        plt.plot(time, experiment, "k-", label="experiment")
        plt.xlabel("time (s)")
        plt.ylabel("norm. fluorescence (a.u.)")
        plt.ylim(top=1)
        plt.legend()

        # add diff_const to image
        plt.annotate(
            rf"D = {diff_const:.3f} $\mu$m$^2$s$^{-1}$",
            xy=(0.65, 0.05),
            xycoords="axes fraction",
        )
        savename_fit = filenames[index].replace(".czi", "_fit.png")
        savename_fit = "".join(["FRAP_analysis/", savename_fit])
        plt.savefig(savename_fit, dpi=300)
        plt.show()

        diffusion_constants.append(diff_const)

    return diffusion_constants


def frap_analysis(
        path: str, bleach_frame: int = 5, recovery_end: int = 100, channel_bleach: int = 0
):
    """
    The frap_analysis function takes a path to a folder of images
    and performs the following:
        1. Initializes data from the images in the folder
        2. Measures fluorescence recovery for each image,
            using an input bleach frame and end frame
        3. Fits data to obtain diffusion constants for each image,
            using Gaussian laser profiles as input

    Args:
        path:str: Specify the location of the data
        bleach_frame:int=5: frame number of bleaching image
        recovery_end:int=100: Specify the end of the recovery period
        channel_bleach:int=0: channel on which bleaching takes place

    Returns:
        A pandas dataframe with the filename, frap position, scaling factor,
         and diffusion constant for each file


    """

    image_data, frap_positions, time_delta, scalings, metadata, filenames = init_data(
        path, bleach_frame, channel_bleach, recovery_end
    )
    # ic(image_data[0].shape)
    time, recovery, radii = measure_fluorescence(
        image_data, frap_positions, time_delta, scalings, metadata, bleach_frame
    )

    diffusion_constants = fit_data(
        time, recovery, radii, filenames, laser_profile="uniform"
    )
    # ic(diffusion_constants)

    # save all relevant info to excel
    df_diffconst = pd.DataFrame(
        {
            "filename": filenames,
            "frap position": frap_positions,
            "scaling": scalings,
            "radii": radii,
            "diffusion constants": diffusion_constants,
        }
    )

    df_diffconst.to_excel("analysis.xlsx", index=False)


# todo: type hints
# todo: background correction (see Kang 2012)
# todo set rood dir for whole package

if __name__ == "__main__":
    # path = input('path to data folder: ')
    DATA_PATH = '../test_data/frap'


    BLEACH_FRAME = 5 # frame just after bleaching
    RECOVERY_END_FRAME = 100
    BLEACH_CHANNEL = 0
    frap_analysis(DATA_PATH, BLEACH_FRAME, RECOVERY_END_FRAME, BLEACH_CHANNEL)
# TODO make local copy of h5 file for better performance and copy at the end to location, delete local copy then