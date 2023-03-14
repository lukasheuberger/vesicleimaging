import sys

from PyQt6.QtWidgets import *
#from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
#from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

import matplotlib.pyplot as plt
import czi_image_handling as cih
import czi_image_analysis as cia
from icecream import ic

# todo make this into a gui wrapper for all other functions

class GUIFunctions():
    """
    Class containing all the helper functions for the GUI
    """

    def __init__(self):
        pass

    def ChooseInputFolder(self):
        """
        Function that opens a file dialog that lets the user choose a file to analyze. It updates the file path in the
        main GUI window and checks the dimensions of the read-in array and updates the amount of measured molos in the
        main GUI window.
        """

        # self.sourcefolder = str(QFileDialog.getExistingDirectory(None, "Select Directory"))
        self.sourcefolder = '/Users/heuberger/code/vesicleimaging/test_data/general'
        print(self.sourcefolder)

        self.FilePathLabel.setText(self.sourcefolder)

        self.OpenFolder()
        self.LoadImageData()
        self.PlotImages()
        # self.PlotImage()

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
# todo make option to convert to hdf5
    def OpenFolder(self):
        print('yay')
        self.files, self.filenames = cih.get_files(self.sourcefolder)
        print(self.files)
        # print (images)
        print('number of images: ', len(self.files))

    def LoadImageData(self):
        self.img_data, self.metadata, self.add_metadata = cih.load_image_data(self.files)
        self.img_reduced = cih.extract_channels(self.img_data)
        ic(len(self.img_reduced))
        ic(self.img_reduced[0].shape)
        # self.channel = self.img[0][1]
        # print(self.img[0].shape)
        # mpl.image.imsave('name.png', self.channel)

    def PlotImages(self):
        for ix, img in enumerate(self.img_reduced):
            cia.plot_images(img, self.metadata[ix], self.add_metadata[ix], saving=True)

    def PlotImage(self):
        # fig, axs = plt.subplots(len(self.img), 3, figsize=(15, 15))
        # # , facecolor='w', edgecolor='k')
        # fig.tight_layout(pad=2)
        # fig.subplots_adjust(hspace=.15, wspace=.2)
        # axs = axs.ravel()
        # print(axs.shape)

        plt.cla()  # clears plot
        self.ax = self.figure.add_subplot(111)

        plt.imshow(self.img[0][0], cmap='gray')
        self.PlotCanvas.draw()

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


class qtGUI(QDialog, GUIFunctions):
    """
    Class containing the GUI for input of data and selection of analysis parameters
    """

    def __init__(self, parent=None):
        """
        Function that initializes the GUI window and creates a box containing all the widgets
        """
        super(qtGUI, self).__init__(parent)
        self.setWindowTitle("CZI Image Handler")
        # self.showMaximized()

        self.createLoadBox()
        # self.createPlotCanvas()
        # self.createOptionBox()

        topLayout = QHBoxLayout()

        mainLayout = QGridLayout()

        mainLayout.addLayout(topLayout, 0, 0)
        mainLayout.addWidget(self.DataLoadBox, 1, 0, 1, 1)
        # mainLayout.addWidget(self.PlotCanvasBox, 1, 1, 4, 1)
        # mainLayout.addWidget(self.OptionBox,            2, 0, 1, 1)

        self.setLayout(mainLayout)

    def createLoadBox(self):
        """
        Function that creates the data loading widget.
        """
        self.DataLoadBox = QGroupBox("Data Selection")
        filePushButton = QPushButton("Load File for Analysis")
        filePushButton.clicked.connect(self.ChooseInputFolder)
        # filePushButton.clicked.connect(self.PlotData)
        # filePushButton.clicked.connect(self.labelsetter)

        self.FilePathLabel = QLabel("no file selected")
        self.FilePathLabel.setWordWrap(True)

        # self.FileShapeLabel = QLabel("")

        layout = QVBoxLayout()
        layout.addWidget(filePushButton)
        layout.addWidget(self.FilePathLabel)
        # layout.addWidget(self.FileShapeLabel)

        self.DataLoadBox.setLayout(layout)

    def createPlotCanvas(self):
        """
        Function that creates a canvas to plot the data and loads a toolbar to edit the graph
        """

        self.PlotCanvasBox = QGroupBox("Plot")
        layout = QVBoxLayout()
        self.PlotCanvasBox.setLayout(layout)

        self.figure = plt.figure(figsize=(3, 2), facecolor='None', edgecolor='None')  # TODO make changeable
        self.PlotCanvas = FigureCanvas(self.figure)
        # set up canvas
        self.toolbarNavigation = NavigationToolbar(self.PlotCanvas, self)
        self.toolbar = QToolBar()

        self.toolbar.addWidget(self.toolbarNavigation)

        layout.addWidget(self.toolbar)

        layout.addWidget(self.PlotCanvas)

    def createOptionBox(self):
        self.OptionBox = QGroupBox("Image Options")

        PlotType_Label = QLabel("Channel:")
        self.PlotType = QComboBox()
        self.PlotType.addItems([0, 1])
        # self.PlotType.currentIndexChanged.connect(self.PlotData)

        layout = QGridLayout()
        layout.addWidget(PlotType_Label, 0, 0)
        layout.addWidget(self.PlotType, 0, 1)

        self.OptionBox.setLayout(layout)

    def createOutputBox(self):
        self.OutputBox = QGroupBox('Output')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    input_window = qtGUI()
    input_window.exec()

# TODO optimize to make faster
