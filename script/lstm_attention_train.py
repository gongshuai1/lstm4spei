import torch
import numpy as np
from torch.autograd import Variable

import argparse
import os
import time
import pickle
import subprocess
import sys

from model.lstm_attention import LSTMAttention
from utils import load_data
from matplotlib import pyplot as plt


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='/home/gongshuai/pythonProject/lstm4spei/data')
    parser.add_argument('--output_path', type=str, default='/home/gongshuai/pythonProject/lstm4spei')
    # RNN size parameter (dimension of the output/hidden state)
    parser.add_argument('--input_size', type=int, default=1)
    parser.add_argument('--output_size', type=int, default=1)
    # RNN size parameter (dimension of the output/hidden state)
    parser.add_argument('--embedding_size', type=int, default=128, help='Embedding dimension for SPEI')
    parser.add_argument('--rnn_size', type=int, default=128, help='size of RNN hidden state')
    parser.add_argument('--heads', type=int, default=8, help='Head number of Multi-head attention')
    parser.add_argument('--depth', type=int, default=6, help='Depth of transformer block')
    parser.add_argument('--mlp_dim', type=int, default=256, help='Dimension of MLP')

    # Size of each batch parameter
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    # Length of sequence to be considered parameter
    parser.add_argument('--seq_length', type=int, default=24, help='RNN sequence length')
    parser.add_argument('--pred_length', type=int, default=12, help='prediction length')

    # Number of epochs parameter
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs')

    # Learning rate parameter
    parser.add_argument('--learning_rate', type=float, default=0.003, help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.95, help='decay rate for rmsprop')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout probability')

    # Lambda regularization parameter (L2)
    parser.add_argument('--lambda_param', type=float, default=0.0005, help='L2 regularization parameter')

    parser.add_argument('--use_cuda', action="store_true", default=False, help='Use GPU or not')

    parser.add_argument('--gru', action="store_true", default=False, help='True : GRU cell, False: LSTM cell')

    parser.add_argument('--num_validation', type=int, default=3,
                        help='Total number of validation dataset for validate accuracy')
    parser.add_argument('--freq_validation', type=int, default=1,
                        help='Frequency number(epoch) of validation using validation data')
    parser.add_argument('--freq_optimizer', type=int, default=8,
                        help='Frequency number(epoch) of learning decay for optimizer')

    # SPEI1, SPEI3, SPEI6, SPEI12, SPEI24, SPEI48
    parser.add_argument('--spei_month_type', type=str, default='48', help='which type SPEI to predict')
    parser.add_argument("--gpu_id", type=str, help="The GPU ID", default='cuda:1')
    # parser.add_argument("--gpu_id", type=str, help="The GPU ID", default='cpu')

    args = parser.parse_args()

    train(args)


def save_checkpoint(args, epoch, model, optimizer):
    save_path = os.path.join(args.output_path, 'checkpoint', f'iteration_SPEI{args.spei_month_type}_{epoch}.pt')
    torch.save(
        {
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'opts': vars(args)
        }, save_path
    )


def r2_loss(y_hat, y):
    target_mean = torch.mean(y)
    ss_tot = torch.sum((y - target_mean) ** 2)
    ss_res = torch.sum((y - y_hat) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


def plot(args, loss_dict, rmse_dict, mae_dict, r2_dict, epoch):
    plt.clf()
    steps = range(len(loss_dict))
    loss = [item.detach().cpu() for item in loss_dict]
    plt.plot(steps, loss)
    plt.title('Loss-steps', fontsize=24)
    plt.xlabel('steps', fontsize=14)
    plt.ylabel('loss', fontsize=14)
    plt.tick_params(axis='both', labelsize=14)
    plt.savefig(os.path.join(args.output_path, 'result/lstm_attention',  f'SPEI{args.spei_month_type}_loss.png'))

    plt.clf()
    steps = range(len(rmse_dict))
    rmse = [item.detach().cpu() for item in rmse_dict]
    plt.plot(steps, rmse)
    plt.title('RMSE-steps', fontsize=24)
    plt.xlabel('steps', fontsize=14)
    plt.ylabel('RMSE', fontsize=14)
    plt.tick_params(axis='both', labelsize=14)
    plt.savefig(os.path.join(args.output_path, 'result/lstm_attention', f'SPEI{args.spei_month_type}_RMSE.png'))

    plt.clf()
    steps = range(len(mae_dict))
    mae = [item.detach().cpu() for item in mae_dict]
    plt.plot(steps, mae)
    plt.title('MAE-steps', fontsize=24)
    plt.xlabel('steps', fontsize=14)
    plt.ylabel('MAE', fontsize=14)
    plt.tick_params(axis='both', labelsize=14)
    plt.savefig(os.path.join(args.output_path, 'result/lstm_attention', f'SPEI{args.spei_month_type}_MAE.png'))

    plt.clf()
    steps = range(len(r2_dict))
    r2 = [item.detach().cpu() for item in r2_dict]
    plt.plot(steps, r2)
    plt.title('R2-steps', fontsize=24)
    plt.xlabel('steps', fontsize=14)
    plt.ylabel('R2', fontsize=14)
    plt.tick_params(axis='both', labelsize=14)
    plt.savefig(os.path.join(args.output_path, 'result/lstm_attention', f'SPEI{args.spei_month_type}_R2.png'))

    loss = torch.tensor(loss, dtype=torch.float32)
    save_excel(args, loss, type='loss')
    rmse = torch.tensor(rmse, dtype=torch.float32)
    save_excel(args, rmse, type='RMSE')
    mae = torch.tensor(mae, dtype=torch.float32)
    save_excel(args, mae, type='MAE')
    r2 = torch.tensor(r2, dtype=torch.float32)
    save_excel(args, r2, type='R2')


def save_excel(args, data, type):
    data = data.numpy()
    save_path = os.path.join(args.output_path, 'result/lstm_attention', f'SPEI{args.spei_month_type}_{type}.xlsx')
    np.savetxt(save_path, data)


def train(args):
    # log

    # dataloader
    stride = args.seq_length // 4
    data_loader = load_data(args.batch_size, args.seq_length, stride, args.spei_month_type, infer=False)
    test_data_loader = load_data(args.batch_size, args.seq_length, args.seq_length, args.spei_month_type, infer=True)

    # model
    model = LSTMAttention(args, infer=False).to(args.gpu_id)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.lambda_param)

    # loss
    loss_fn = torch.nn.MSELoss()
    r2_fn = r2_loss
    loss_dict = []
    # validate metric
    rmse_dict = []
    mae_dict = []
    r2_dict = []

    # Training
    for epoch in range(args.num_epochs):
        print('****************Training epoch beginning******************')
        loss_epoch = 0.0

        begin = time.time()
        for batch_id, batch in enumerate(data_loader):
            batch = batch.to(args.gpu_id)
            x = torch.unsqueeze(batch, dim=2)
            x = x.to(args.gpu_id)
            # forward
            y_hat = torch.squeeze(model(x))

            # compute loss
            loss = loss_fn(batch, y_hat)
            loss_epoch = (loss_epoch * batch_id + loss.detach()) / (batch_id + 1)

            if torch.isnan(loss_epoch):
                print(f'batch_id = {batch_id}')
                print(f'batch = {batch}')
                print(f'y_hat = {y_hat}')
                print(f'loss = {loss}')
                sys.exit()

            # back forward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        end = time.time()

        loss_dict.append(loss_epoch)

        print(f'{epoch}th epoch training, {end - begin} seconds, loss = {loss_epoch}')

        # Validate
        rmse_epoch = 0.0
        mae_epoch = 0.0
        r2_epoch = 0.0
        for batch_id, batch in enumerate(test_data_loader):
            batch = batch.to(args.gpu_id)
            x = torch.unsqueeze(batch, dim=2)
            x = x.to(args.gpu_id)
            # forward
            y_hat = torch.squeeze(model(x))

            rmse = torch.pow(loss_fn(batch, y_hat), 0.5)
            rmse_epoch = (rmse_epoch * batch_id + rmse.detach()) / (batch_id + 1)
            mae = torch.abs(batch - y_hat).mean()
            mae_epoch = (mae_epoch * batch_id + mae.detach()) / (batch_id + 1)
            r2 = r2_fn(y_hat, batch)
            r2_epoch = (r2_epoch * batch_id + r2.detach()) / (batch_id + 1)

        rmse_dict.append(rmse_epoch)
        mae_dict.append(mae_epoch)
        r2_dict.append(r2_epoch)

    # Save the model after each epoch
    print('Saving model')
    save_checkpoint(args, epoch, model, optimizer)
    # plot
    plot(args, loss_dict, rmse_dict, mae_dict, r2_dict, epoch)


if __name__ == '__main__':
    main()
