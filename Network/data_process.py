import os
import numpy as np
import pandas as pd
import scipy.io as sio
import torch

from torch.utils.data import DataLoader, Dataset


def z_score(dataset):
    mean = np.mean(dataset)
    std = np.std(dataset)
    z_data = (dataset - mean) / std
    return z_data

'数据读取'
class CreateDataset(Dataset):
    def __init__(self, folders, root_dir, filename, mode, usage, thresh=0, extension='.mat'):
        self.data_paths = []
        self.extension = extension
        self.mode = mode
        self.usage = usage
        self.thresh = thresh

        for folder in folders:
            data_path = os.path.join(root_dir, folder, filename)
            for data in os.listdir(data_path):
                self.data_paths.append(os.path.join(data_path, data))

        self.data_len = len(self.data_paths)


    def __getitem__(self, index):
        idx_data = self.data_paths[index]
        dataset = sio.loadmat(idx_data)
        data = {}
        # 同时利用CSI和谱信息
        if self.mode == 'csi_spectrum':
            amp_ = z_score(dataset['amp']).reshape(6, 6)
            phase_ = z_score(dataset['phase']).reshape(6, 6)

            sp = z_score(dataset['spectrum'])

            csi = np.stack([amp_, phase_])

            csi = torch.from_numpy(csi).float()
            sp = torch.from_numpy(sp).float().unsqueeze(0)
            if torch.isnan(csi).int().sum() > 0:
                csi = torch.where(torch.isnan(csi), torch.tensor(0), csi)
            if torch.isnan(sp).int().sum() > 0:
                sp = torch.where(torch.isnan(sp), torch.tensor(0), sp)
            data['csi'] = csi
            data['sp'] = sp


        if self.usage == 'train':
            label = dataset['rmse'] <= self.thresh
            label = torch.tensor(label, dtype=torch.int64).squeeze()

            return (data, label)
        elif self.usage == 'test':
            return (data)

    def __len__(self):
        return self.data_len


# 生成训练集
def get_loaders(train_csv, val_csv, root_dir, filename, mode, thresh=0, batch_size=64):
    dataloaders = {}
    if train_csv == '':
        dataloaders['train'] = []
    else:
        train_df = pd.read_csv(train_csv)
        train_files = train_df['path'].tolist()
        train_dataset = CreateDataset(train_files, root_dir, filename, mode, 'train', thresh)
        dataloaders['train'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    if val_csv == '':
        dataloaders['val'] = []
    else:
        val_df = pd.read_csv(val_csv)
        val_files = val_df['path'].tolist()
        val_dataset = CreateDataset(val_files, root_dir, filename, mode, 'train')
        dataloaders['val'] = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    return dataloaders['train'], dataloaders['val']


def get_test_loaders(test_file, root_dir, filename, mode):
    dataset = CreateDataset(test_file, root_dir, filename, mode, 'test')
    dataloaders = DataLoader(dataset, batch_size=len(dataset))
    return dataloaders


if __name__ == '__main__':
    root_dir = 'E:/bupt/mmWave/trainsets'
    filename = 'thresh5_win50'
    train_csv = 'train_list.csv'
    val_csv = 'test_list.csv'
    train_loader, val_loader = get_loaders(train_csv, val_csv, root_dir, filename, 'csi_spectrum', thresh=1)
    data_path = '../../Data/trainsets/train/'
