import time
import numpy as np
import argparse

# Arguments : Need to be written
parser = argparse.ArgumentParser()
parser.add_argument('--csv-route', type=str, default='data/original_dataset.csv', help='Original csv file containing close price.')

args = parser.parse_args()

def load_csvfile(csv_route):
    csv_file = np.loadtxt(csv_route, delimiter=',', dtype=np.float32)

def generate_datset(num_sims):
    close_all = list()

    for i in range(num_sims):
        t = time.time()

