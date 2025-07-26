import csv
import os.path
import time
import torch
import pickle

from tqdm import tqdm
from train_func import train_model, val_model
from data_process import get_loaders
from classifier import CSISPClassifier

import warnings
warnings.filterwarnings("ignore")


'训练CSI-SP组合模型'
def train(model, train_loader, val_loader, model_path, criterion, device, lr=0.001, epochs=20):
    iteration = 0
    time_start = time.time()
    header = ['epoch', 'learning rate', 'train loss', 'train acc', 'val loss', 'val acc', 'TP', 'TN', 'FP', 'FN', 'time cost now/second']

    model = model.to(device)
    model.train()
    model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count()))).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = 1000000

    for epoch in tqdm(range(epochs)):
        if epoch > 5:
            if lr > 0.0001:
                lr = lr * 0.5
        print('\nLearning Rate = ', round(lr,6), end='\n')
        train_loss, train_acc, iteration = train_model(model, device, train_loader, optimizer, criterion, iteration)
        val_loss, val_acc, TP, TN, FP, FN = val_model(model, device, val_loader, criterion)
        time_cost_now = time.time() - time_start

        values = [epoch + 1, lr, train_loss, train_acc, val_loss, val_acc, TP, TN, FP, FN, time_cost_now]

        if not os.path.exists(model_path):
            os.makedirs(model_path)
        path_csv = model_path + "loss and others" + ".csv"
        if os.path.isfile(path_csv) == False:
            file = open(path_csv, 'w', newline='')
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerow(values)
        else:
            file = open(path_csv, 'a', newline='')
            writer = csv.writer(file)
            writer.writerow(values)
        file.close()
        # Save model
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'loss': train_loss,
                'optimizer': optimizer.state_dict(),
            }, model_path + "weights" + ".pth")
        torch.save({'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
            }, model_path + "test_weights_epoch_" + str(epoch + 1) + ".pth")


if __name__ == '__main__':
    torch.cuda.empty_cache()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model_path = 'tosn_model/csisp_20/'

    root_dir = 'E:/mmWave/trainsets'
    filename = 'thresh5_win50'
    train_csv = 'train_list.csv'
    val_csv = 'test_list.csv'

    # 读取数据内容
    train_loader, val_loader = get_loaders(train_csv, val_csv, root_dir, filename, 'csi_spectrum')

    criterion = torch.nn.BCELoss()
    # 定义模型
    model = CSISPClassifier()
    train(model, train_loader, val_loader, model_path, criterion, device)
    with open(model_path+'model.pkl', 'wb') as f:
        pickle.dump(model, f)