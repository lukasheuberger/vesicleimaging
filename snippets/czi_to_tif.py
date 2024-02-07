import czifile
import tifffile
import os

# Directory containing the .czi files
root_directory = '/Users/heuberger/Desktop/zenodo_data/fig6/6e_FRAP/'

# Walk through the directory
for subdir, dirs, files in os.walk(root_directory):
    for filename in files:
        if filename.endswith('.czi'):
            # Construct the full file path
            file_path = os.path.join(subdir, filename)

            # Read the .czi file
            with czifile.CziFile(file_path) as czi:
                image = czi.asarray()

            # Construct the output .tif file name
            tif_filename = os.path.splitext(filename)[0] + '.tif'
            tif_path = os.path.join(subdir, tif_filename)

            # Write the .tif file
            tifffile.imwrite(tif_path, image)


print("Conversion complete.")