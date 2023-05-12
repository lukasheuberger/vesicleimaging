from fcsfiles import ConfoCor3Fcs
import pandas as pd


def get_fcs_run_data(fcs_file):

    fcs_data = ConfoCor3Fcs(fcs_file)

    #print(f'number of runs: {len(fcs_file['FcsData']['FcsEntry'])}')
    #print(f'number of runs: {len(fcs_data['FcsData'])}')
    #test = fcs_data['FcsData']['FcsEntry']
    print(f'number of runs (incl. average): {len(fcs_data["FcsData"]["FcsEntry"])}')

    # Create an empty DataFrame
    df = pd.DataFrame()

    for index, rep in enumerate(fcs_data["FcsData"]["FcsEntry"][0:-1]):
        run_data = rep["FcsDataSet"]["CorrelationArray"]

        #print(run_data)

        # Assuming run_data is a 2D array and each nested array has 2 elements
        # Add them as new columns to the DataFrame
        data = []
        for i, d in enumerate(run_data):
            #print(data[1])
            data.append(d[1])

        df[index+1] = data # make it not zero indexed
    df['average'] = df.mean(axis=1)
    df['stdev'] = df.std(axis=1)
    # print(df)

    return df
