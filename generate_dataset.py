import time
import numpy as np
import pandas as pd
import argparse

# Arguments : Need to be written
parser = argparse.ArgumentParser()
parser.add_argument('--csv-route', type=str, default='data/original_dataset.csv', help='Original csv file containing close price.')

args = parser.parse_args()

def load_csvfile(csv_route):
    csv_file = pd.read_csv(csv_route)
    return csv_file

def generate_dataset(csv_route, num_sims, length, asset_names):
    # csv_file : Column name need to be defined
    csv_file = load_csvfile(csv_route)

    if num_sims+length>csv_file.shape[0]:
        print("Number of simulation {} exceeded the maximum length {} of dataset.".format(num_sims, length))
    else:
        close_all = list()
        data_all = dict()
        for name in asset_names:
            data_ = csv_file.loc[csv_file['TICKER'] == name][["TICKER", "date", "PRC"]]
            data_all[name] = data_

        for i in range(num_sims):
            t = time.time()
            close_ = list()
            for name in asset_names:
                close = data_all[name].iloc[i:i+length]["PRC"].to_numpy()
                close_.append(close)
            close_ = np.stack(close_)
            close_all.append(close_)
        close_all = np.stack(close_all)

        return close_all

if __name__ == "__main__":
    csv_route = "data/data.csv"
    num_sims = 1000
    length = 5
    asset_names = ["INTC", "AMD", "KO", "PEP"]
    price_train = generate_dataset(csv_route, num_sims, length, asset_names)

    print("Generating simulations")
    np.save('data/price_train.npy', price_train)

