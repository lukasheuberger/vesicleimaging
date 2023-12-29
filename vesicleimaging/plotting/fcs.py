# Description: Functions for reading and processing FCS files

from fcsfiles import ConfoCor3Fcs
import pandas as pd


def get_fcs_run_data(fcs_file, average=False):
    """
    The get_fcs_run_data function takes a single FCS file and returns a pandas DataFrame with the following columns:
        - Columns 1-n are the individual runs (incl. average)
        - 'average' is the mean of all runs (excluding average)
        - 'stddev' is the standard deviation of all runs (excluding average)

    Args:
        fcs_file: Specify the file that is being read

    Returns:
        A dataframe with the data from each run and the average
    """

    fcs_data = ConfoCor3Fcs(fcs_file)
    print(f'number of runs (incl. average): {len(fcs_data["FcsData"]["FcsEntry"])}')

    # Create an empty DataFrame
    df = pd.DataFrame()

    for index, rep in enumerate(fcs_data["FcsData"]["FcsEntry"][0:-1]):
        run_data = rep["FcsDataSet"]["CorrelationArray"]

        # Assuming run_data is a 2D array and each nested array has 2 elements
        # Add them as new columns to the DataFrame
        data = []
        for ix, rep_data in enumerate(run_data):
            data.append(rep_data[1])

        df[index+1] = data  # make it not zero indexed
    if average:
        df['average'] = df.mean(axis=1)
        df['stddev'] = df.std(axis=1)

    return df
