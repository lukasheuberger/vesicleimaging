import os
import sys

import matplotlib.pyplot as plt
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import *
from icecream import ic
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib_scalebar.scalebar import ScaleBar

# import czi_image_analysis as cia
# import czi_image_handling as cih

# todo make this into a gui wrapper for all other functions

class GUIFunctions():
    """
    Class containing all the helper functions for the GUI
    """

    def __init__(self):
        pass

    def ChooseInputCZIFolder(self):
        """
        Function that opens a file dialog that lets the user choose a file to analyze. It updates the file path in the
        main GUI window and checks the dimensions of the read-in array and updates the amount of measured molos in the
        main GUI window.
        """

        # self.sourcefolder = str(QFileDialog.getExistingDirectory(None, "Select Directory"))
        self.sourcefolder = '/Users/lukasheuberger/code/phd/vesicle-imaging/test_data/general'
        print(self.sourcefolder)
        self.filetype = 'czi'
        self.Initialize()

        # if self.fileNameData:
        #     print(self.fileNameData)
        #     if self.fileType == "db Files (*.db)":
        #         self.data_path, tail = os.path.split(self.fileNameData)
        #         sys.path.append(self.data_path)
        #         os.chdir(self.data_path)
        #         self.data_raw = self.load_column_from_table(self.fileNameData, 'default_real_time_structure_accP_wg_OCP', 'Gamma_coh_1')
        #         self.time = self.load_column_from_table(self.fileNameData, "default_real_time_structure_acc", "time_stamp_s")
        #         self.data_rows = len(self.data_raw)
        #         self.data_cols = 1
        #     else:
        #         self.data_path, tail = os.path.split(self.fileNameData)
        #         sys.path.append(self.data_path)
        #         os.chdir(self.data_path)
        #         self.data_raw = pd.read_csv(self.fileNameData, sep=",", header=None)
        #         self.data_rows, self.data_cols = self.data_raw.shape
        #         self.data_cols = self.data_cols - 1
        #
        #         for i in range(30 - self.data_cols):
        #             # disables selected molos that are out of range
        #             # and would crash the script
        #             if i < 10:
        #                 self.MoloLine3[9 - i].setEnabled(False)
        #                 self.MoloLine3[9 - i].setChecked(False)
        #             if i < 20 and i > 9:
        #                 self.MoloLine2[9 - i].setEnabled(False)
        #                 self.MoloLine2[9 - i].setChecked(False)
        #             if i < 30 and i > 19:
        #                 self.MoloLine1[9 - i].setEnabled(False)
        #                 self.MoloLine1[9 - i].setChecked(False)
        #
        #     print(self.data_rows)#, self.data_cols)
        #     self.SignalBox_TruncateAfter.setValue(self.data_rows)
        #     self.SignalBox_TruncateBefore.setMaximum(self.data_rows)
        #
        # self.plot_params_file = open("plot_parameters.txt", "w")
        #
        # no_molos = "".join(("number of molos measured: ", str(self.data_cols)))
        #
        # self.FileShapeLabel.setText(no_molos)

    def Initialize(self):
        os.chdir(self.sourcefolder)
        print(self.sourcefolder)

        self.FilePathLabel.setText(self.sourcefolder)

        self.OpenFolder()
        self.LoadImageData()
        # self.SaveImages()
        self.PlotImage()

    # todo determine somewhere if data is czi or hdf5 and act accordingly (already does it but would like to be more elegant)
    def ChooseInputHDF5Folder(self):
        # self.sourcefolder = str(QFileDialog.getExistingDirectory(None, "Select Directory"))
        self.sourcefolder = '/Users/lukasheuberger/code/phd/vesicle-imaging/test_data/general'
        print(self.sourcefolder)
        self.filetype = 'hdf5'
        self.Initialize()

    def OpenFolder(self):
        self.files, self.filenames = cih.get_files(self.sourcefolder)
        ic(self.files)
        # print (images)
        print('number of images: ', len(self.files))

    def WriteMetadataToXML(self):
        cih.write_metadata_xml(self.sourcefolder, self.files)

    def LoadChannels(self, index=0):
        index = self.ImageSelector.currentIndex()
        try:
            # for ix, amd in enumerate(self.add_metadata[0]):
            channels = cih.get_channels([self.add_metadata[index]])  # this also needs to change when image changes

            self.channel_names = []
            for channel in channels[2]:
                if channel is None:
                    self.channel_names.append('T-PMT')
                elif channel == "BODIPY 630/650-X":
                    self.channel_names.append('BODIPY 630-650-X')
                else:
                    self.channel_names.append(channel.replace(" ", ""))

            ic(self.channel_names)

            self.PlotChannel.clear()
            self.PlotChannel.addItems(self.channel_names)

            self.DetectDisplayChannel.clear()
            self.DetectDisplayChannel.addItems(self.channel_names)

            self.DetectDetectChannel.clear()
            self.DetectDetectChannel.addItems(self.channel_names)

        except AttributeError:
            pass

    def LoadImageData(self):

        if self.filetype == 'czi':
            self.img_data, self.metadata, self.add_metadata = cih.load_image_data(self.files)
            self.img_reduced = cih.extract_channels(self.img_data)

        if self.filetype == 'hdf5':
            self.img_reduced, self.metadata, self.add_metadata = cih.load_h5_data(self.sourcefolder)

        ic(len(self.img_reduced))
        ic(self.img_reduced[0].shape)

        self.filenames = []

        for ix, md in enumerate(self.metadata):
            try:
                filename = md['Filename'].replace('.czi', '')
            except TypeError:
                filename = md[0]['Filename'].replace('.czi', '')

            self.filenames.append(filename)

        ic(self.filenames)

        self.ImageSelector.addItems(self.filenames)

        self.LoadChannels()  # index=0)

        # todo probably some of this can be their own functions

        self.DetermineImageType()

        self.MetadataWriteButton.setEnabled(True)

    def DetermineImageType(self):
        self.image_type = 'standard'

        self.Timepoint_Slider.setValue(0)
        self.Timepoint_Slider.setEnabled(False)
        # todo make that it doesn't crash when timepoint slider is not zero and switch to other image type
        self.ZPosition_Slider.setValue(0)
        self.ZPosition_Slider.setEnabled(False)

        if self.img_reduced[self.ImageSelector.currentIndex()].shape[1] > 1:
            self.image_type = 'timelapse'
            print('timelapse!')

            self.timelapse_frames = self.img_reduced[self.ImageSelector.currentIndex()].shape[1]
            self.Timepoint_Slider.setMaximum(self.timelapse_frames - 1)
            self.Timepoint_Slider.setEnabled(True)

        if self.img_reduced[self.ImageSelector.currentIndex()].shape[2] > 1:
            self.image_type = 'z-stack'
            print('z-stack!')

            self.zstack_slices = self.img_reduced[self.ImageSelector.currentIndex()].shape[2]
            self.ZPosition_Slider.setMaximum(self.zstack_slices - 1)
            self.ZPosition_Slider.setEnabled(True)

        self.ImageType.setText(self.image_type)

    def SaveImagesToPNG(self):
        for ix, img in enumerate(self.img_reduced):
            cia.plot_images(img, self.metadata[ix], self.add_metadata[ix], saving=True, display=False)

    # todo make possible to load from HDF5
    def SaveImagesToHDF5(self):
        cih.save_files(self.img_reduced, self.metadata, self.add_metadata)

    # todo make possible to save xml metadata

    def PlotImage(self):
        # ic(self.PlotChannel.currentText())
        # ic(self.ImageSelector.currentText())
        # ic(self.ImageSelector.currentIndex())
        # ic(self.Timepoint_Slider.sliderPosition())

        try:
            temp_filename = self.metadata[0]['Filename'].replace('.czi', '')
        except TypeError:
            temp_filename = self.metadata[0][0]['Filename'].replace('.czi', '')

        ic(temp_filename)

        scaling_x = cih.disp_scaling([self.add_metadata[self.ImageSelector.currentIndex()]])
        # ic(scaling_x)

        plt.cla()  # clears plot
        self.ax = self.figure.add_subplot(111, frameon=False)
        plt.tight_layout(pad=0)
        self.ax.set_title(temp_filename)
        # self.ax.axes.xaxis.set_ticklabels([])
        # self.ax.axes.yaxis.set_ticklabels([])
        # plt.axis('off')
        # if scalebar:  # 1 pixel = scale [m]
        scalebar_value = 50
        scalebar = ScaleBar(dx=scaling_x[0], location='lower right', fixed_value=scalebar_value, fixed_units='µm',
                            frameon=False, color='w')

        ic(self.img_reduced[self.ImageSelector.currentIndex()].shape)

        # self.ax = self.PlotCanvas.figure.subplots()
        self.currentImage = self.img_reduced[self.ImageSelector.currentIndex()]

        self.ax.imshow(self.currentImage[self.PlotChannel.currentIndex()][self.Timepoint_Slider.sliderPosition()][self.ZPosition_Slider.sliderPosition()], cmap=self.CmapSelector.currentText())
        self.ax.add_artist(scalebar)
        # self.ax.set_axis_off() somehow doesn't work

        # hide x-axis
        self.ax.get_xaxis().set_visible(False)
        # hide y-axis
        self.ax.get_yaxis().set_visible(False)

        self.PlotCanvas.draw()

    def PlotDetectedCircles(self):
        plt.imshow(self.output_circle_img)
        plt.show()

        # subfig_counter = 0
        # filename_counter = 0

        # for image in self.img:
        # print ('image:',img)
        # print(self.metadata[filename_counter]['Filename'])
        # for channel in range(0, len(image)):
        # print ('channel:',channel)

        # print(filename_counter)
        # if filename_counter < 1: #so only top two images have channel names
        #    axs[subfig_counter].title.set_text(channel_names[channel])
        # if saving_on == True:
        #    temp_filename = self.metadata[filename_counter]['Filename'].replace('.czi', '')
        #    output_filename = ''.join(['analysis/', temp_filename, '_', str(channel + 1), '.png'])
        #    # print(output_filename)
        #    plt.imsave(output_filename, image[channel], cmap='gray')

        # subfig_counter = subfig_counter + 1
        # filename_counter = filename_counter + 1

    def updateSliderLabel(self, value):
        #todo make all this naming consistent
        if self.sender() == self.Timepoint_Slider:
            self.TimeSliderTimepoint.setText(str(value))
        elif self.sender() == self.ZPosition_Slider:
            self.ZStackSliderPosition.setText(str(value))
        elif self.sender() == self.Param1_Slider:
            self.Param1_SliderValue.setText(str(value))
        elif self.sender() == self.Param2_Slider:
            self.Param2_SliderValue.setText(str(value))

    def detectGUVs(self):

        #todo make this work for multiple images

        self.detected_circles, self.output_circle_img = cia.detect_circles([self.currentImage], [self.metadata[self.ImageSelector.currentIndex()]],
                                          param1_array=self.Param1_Slider.sliderPosition(),
                                          param2_array=self.Param2_Slider.sliderPosition(),
                                          minmax=[int(self.SizeMin.text()), int(self.SizeMax.text())],
                                          display_channel=self.DetectDisplayChannel.currentIndex(),
                                          detection_channel=self.DetectDetectChannel.currentIndex(),
                                                                           plot=False, return_image=True)

        self.PlotDetectedCircles()


class qtGUI(QDialog, GUIFunctions):
    """
    Class containing the GUI for input of data and selection of analysis parameters
    """

    def __init__(self, parent=None):
        """
        Function that initializes the GUI window and creates a box containing all the widgets
        """
        super(qtGUI, self).__init__(parent)
        self.setWindowTitle("GUV Image Handler")
        # self.showMaximized()

        self.createLoadBox()
        self.createSaveBox()
        self.createPlotCanvas()
        self.createImageSelectionBox()
        self.createOptionBox()
        self.createPlotOptionBox()
        self.createDetectCircleBox()

        topLayout = QHBoxLayout()

        mainLayout = QGridLayout()

        mainLayout.addLayout(topLayout, 0, 0)
        mainLayout.addWidget(self.DataLoadBox, 1, 0, 1, 1)
        mainLayout.addWidget(self.SaveBox, 2, 0, 1, 1)
        mainLayout.addWidget(self.PlotCanvasBox, 1, 1, 4, 1)
        mainLayout.addWidget(self.ImageSelectionBox, 3, 0, 1, 1)
        mainLayout.addWidget(self.OptionBox, 4, 0, 1, 1)
        mainLayout.addWidget(self.PlotOptionBox, 5, 1, 1, 1)
        mainLayout.addWidget(self.DetectCircleBox, 5, 0, 1, 1)

        self.setLayout(mainLayout)

    def createLoadBox(self):
        """
        Function that creates the data loading widget.
        """
        self.DataLoadBox = QGroupBox("Data Selection")
        cziLoadButton = QPushButton("Load folder containing CZI images for analysis")
        cziLoadButton.clicked.connect(self.ChooseInputCZIFolder)

        self.FilePathLabel = QLabel("no folder selected")
        self.FilePathLabel.setWordWrap(True)

        HDF5LoadButton = QPushButton('Load folder containing HDF5 files for analysis (faster)')
        HDF5LoadButton.clicked.connect(self.ChooseInputHDF5Folder)

        self.MetadataWriteButton = QPushButton('write metadata to XML dict in source folder')
        self.MetadataWriteButton.setEnabled(False)
        self.MetadataWriteButton.clicked.connect(self.WriteMetadataToXML)

        # cziLoadButton.clicked.connect(self.PlotData)
        # cziLoadButton.clicked.connect(self.labelsetter)

        # self.FileShapeLabel = QLabel("")

        layout = QVBoxLayout()
        layout.addWidget(cziLoadButton)
        layout.addWidget(HDF5LoadButton)
        layout.addWidget(self.FilePathLabel)
        layout.addWidget(self.MetadataWriteButton)

        # layout.addWidget(self.FileShapeLabel)

        self.DataLoadBox.setLayout(layout)

    def createSaveBox(self):
        # todo only make this accessible if data is loaded
        self.SaveBox = QGroupBox('Data Handling')

        fileSaveButton = QPushButton("Save Images to png's")
        fileSaveButton.clicked.connect(self.SaveImagesToPNG)

        fileToHDF5Button = QPushButton("Convert Images to HDF5 files")
        fileToHDF5Button.clicked.connect(self.SaveImagesToHDF5)

        layout = QVBoxLayout()
        layout.addWidget(fileSaveButton)
        layout.addWidget(fileToHDF5Button)

        self.SaveBox.setLayout(layout)

    def createPlotCanvas(self):
        """
        Function that creates a canvas to plot the data and loads a toolbar to edit the graph
        """

        self.PlotCanvasBox = QGroupBox("Plot")
        layout = QVBoxLayout()
        self.PlotCanvasBox.setLayout(layout)

        self.figure = plt.figure(figsize=(2, 2), facecolor='None', edgecolor='None',
                                 frameon=False)  # TODO make changeable
        # self.figure.axis('off')

        self.PlotCanvas = FigureCanvas(self.figure)
        # set up canvas
        self.toolbarNavigation = NavigationToolbar(self.PlotCanvas, self)
        self.toolbar = QToolBar()

        self.toolbar.addWidget(self.toolbarNavigation)

        layout.addWidget(self.toolbar)

        layout.addWidget(self.PlotCanvas)

    def createImageSelectionBox(self):
        self.ImageSelectionBox = QGroupBox('Image Selection')

        ImageSelector_Label = QLabel('Select Image')
        self.ImageSelector = QComboBox()
        # self.ImageSelector.addItems(['0', '1'])
        # self.ImageSelector.currentIndexChanged.connect(self.LoadChannels(index=1))
        self.ImageSelector.currentIndexChanged.connect(self.LoadChannels)
        self.ImageSelector.currentIndexChanged.connect(self.DetermineImageType)
        self.ImageSelector.currentIndexChanged.connect(self.PlotImage)

        layout = QGridLayout()
        layout.addWidget(ImageSelector_Label, 0, 0)
        layout.addWidget(self.ImageSelector, 1, 0)

        self.ImageSelectionBox.setLayout(layout)

    def createOptionBox(self):
        self.OptionBox = QGroupBox("Image Options")

        PlotChannel_Label = QLabel("Displayed Channel:")
        self.PlotChannel = QComboBox()
        self.PlotChannel.addItems(['no image selected'])
        self.PlotChannel.currentIndexChanged.connect(self.PlotImage)

        CmapSelector_Label = QLabel('Colormap:')
        self.CmapSelector = QComboBox()
        # print(plt.colormaps)
        self.CmapSelector.addItems(plt.colormaps())
        self.CmapSelector.setCurrentIndex(plt.colormaps().index('gray'))
        self.CmapSelector.currentIndexChanged.connect(self.PlotImage)

        layout = QGridLayout()
        layout.addWidget(PlotChannel_Label, 0, 0)
        layout.addWidget(self.PlotChannel, 0, 1)

        layout.addWidget(CmapSelector_Label, 1, 0)
        layout.addWidget(self.CmapSelector, 1, 1)

        self.OptionBox.setLayout(layout)

    def createOutputBox(self):
        self.OutputBox = QGroupBox('Output')

    def createPlotOptionBox(self):
        self.PlotOptionBox = QGroupBox('Plot Options')

        ImageTypeLabel = QLabel('Image Type:')
        self.ImageType = QLabel('')

        # todo display image type somewhere here
        # todo only show this if image is actually timelapse
        TimeSliderLabel = QLabel('timepoint')
        self.Timepoint_Slider = QSlider(Qt.Orientation.Horizontal, self)
        self.Timepoint_Slider.setTickInterval(1)
        self.Timepoint_Slider.valueChanged.connect(self.updateSliderLabel)
        self.Timepoint_Slider.valueChanged.connect(self.PlotImage)
        # todo set slider position to middle and also disp middle image
        self.TimeSliderTimepoint = QLabel()

        ZPositionSiderLabel = QLabel('z-position')
        self.ZPosition_Slider = QSlider(Qt.Orientation.Horizontal, self)

        self.ZPosition_Slider.valueChanged.connect(self.PlotImage)
        self.ZStackSliderPosition = QLabel()

        layout = QGridLayout()
        layout.addWidget(ImageTypeLabel, 0, 0)
        layout.addWidget(self.ImageType, 0, 1)
        layout.addWidget(TimeSliderLabel, 1, 0)
        layout.addWidget(self.Timepoint_Slider, 1, 1)
        layout.addWidget(self.TimeSliderTimepoint, 1, 2)

        layout.addWidget(ZPositionSiderLabel, 2, 0)
        layout.addWidget(self.ZPosition_Slider, 2, 1)
        layout.addWidget(self.ZStackSliderPosition, 2, 2)

        self.PlotOptionBox.setLayout(layout)

    def createDetectCircleBox(self):
        self.DetectCircleBox = QGroupBox('Detect GUVs')

        Param_Instructions = QLabel('Adjust the parameters to detect GUVs \n'
                                    'the bigger param1, the fewer circles may be detected \n'
                                    'the smaller param2, the more false circle sare detected')

        # todo sync this with a text box
        Param1SliderLabel = QLabel('param1')
        self.Param1_Slider = QSlider(Qt.Orientation.Horizontal, self)
        self.Param1_Slider.setRange(1, 200)
        self.Param1_Slider.setValue(10)
        self.Param1_Slider.setTickInterval(1)
        self.Param1_Slider.valueChanged.connect(self.updateSliderLabel)
        self.Param1_SliderValue = QLabel()

        Param2SliderLabel = QLabel('param2')
        self.Param2_Slider = QSlider(Qt.Orientation.Horizontal, self)
        self.Param2_Slider.setRange(1, 250)
        self.Param2_Slider.setValue(150)
        self.Param2_Slider.setTickInterval(1)
        self.Param2_Slider.valueChanged.connect(self.updateSliderLabel)
        self.Param2_SliderValue = QLabel()

        SizeLabel = QLabel('GUV size [px] (min, max)')
        self.SizeMin = QLineEdit('45')
        self.SizeMax = QLineEdit('60')

        DisplayChannelLabel = QLabel('Display Channel')
        self.DetectDisplayChannel = QComboBox()

        DetectChannelLabel = QLabel('Detection Channel')
        self.DetectDetectChannel = QComboBox()

        self.DetectCircleButton = QPushButton('Detect GUVs on current image')
        self.DetectCircleButton.clicked.connect(self.detectGUVs)

        self.DetectAllCirclesButton = QPushButton('Detect GUVs on all images')

        layout = QGridLayout()
        layout.addWidget(Param_Instructions, 0, 0)

        layout.addWidget(Param1SliderLabel, 1, 0)
        layout.addWidget(self.Param1_Slider, 1, 1)
        layout.addWidget(self.Param1_SliderValue, 1, 2)

        layout.addWidget(Param2SliderLabel, 2, 0)
        layout.addWidget(self.Param2_Slider, 2, 1)
        layout.addWidget(self.Param2_SliderValue, 2, 2)

        layout.addWidget(SizeLabel, 3, 0)
        layout.addWidget(self.SizeMin, 3, 1)
        layout.addWidget(self.SizeMax, 3, 2)

        layout.addWidget(DisplayChannelLabel, 4, 0)
        layout.addWidget(self.DetectDisplayChannel, 4, 1)

        layout.addWidget(DetectChannelLabel, 5, 0)
        layout.addWidget(self.DetectDetectChannel, 5, 1)

        layout.addWidget(self.DetectCircleButton, 6, 0)
        layout.addWidget(self.DetectAllCirclesButton, 6, 1)

        self.DetectCircleBox.setLayout(layout)


if __name__ == '__main__':
    vi.test()
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    input_window = qtGUI()
    input_window.exec()
