import numpy as np
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable

def load_data(batch_size=1):
    price_train = np.load('data/price_train.npy')
    blank = np.full_like(price_train, 0)

    num_assets = price_train.shape[1]
    
    # skip normalization
    price_train = torch.FloatTensor(price_train)
    blank = torch.FloatTensor(blank)
    
    train_data = TensorDataset(price_train, blank)
    train_data_loader = DataLoader(train_data, batch_size=batch_size)

    return train_data_loader

if __name__ == "__main__":
    a = load_data()
    print()