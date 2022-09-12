from vesicle_imaging import czi_image_handling as handler
from icecream import ic

def write_metadata(path):
    # os.listdir(path)
    # curr_dir = os.getcwd()
    # os.chdir(path)

    files, filenames = handler.get_files(path)
    filenames.sort(reverse=True)
    files.sort(reverse=True)
    print (files)

    #print (images)
    print ('number of images: ', len(files))

    print (files)
    #filenames = files
    #filenames.sort()
    for i in range(0,len(files)):
        filenames[i] = files[i].replace('.czi','')
    ic(filenames)

    # image_data = []
    # image_metadata = []
    # image_add_metadata = []
    # for image in files:
    #     print (image)
    #     data, metadata, add_metadata = handler.load_image_data([image])
    #     image_data.append(data)
    #     image_metadata.append(metadata)
    #     image_add_metadata.append(add_metadata)
    # print ('IMAGE IMPORT DONE!')
    # os.chdir(curr_dir)

    # image_xy_data = handler.extract_channels_xy(image_data)
    # image_xy_data[0][0].shape

    # handler.disp_basic_img_info(image_xy_data, image_metadata)

    #handler.disp_all_metadata(image_metadata)
    #channels = handler.disp_channels(image_add_metadata, type = 'MultiTrack')

    handler.write_metadata_xml(path, files)

if __name__ == "__main__":
    path = input('path to files to convert: ')
    print('path: ', path)
    write_metadata(path)
    print('xml metadata written')