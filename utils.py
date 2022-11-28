import torch
import torch.utils.data
from dataset.SPEIDataset import SPEIDataset


def load_data(batch_size=32, seq_length=20, stride=1, month_type=1, infer=False):
    dataset = SPEIDataset(seq_length, stride, month_type, infer)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return data_loader
