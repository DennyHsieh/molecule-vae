import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader, random_split
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from rdkit import Chem


def train(model, train_loader, optimizer, loss_func, epoch):
    model.train()
    cnt = 0
    loss_total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target.reshape(-1, 1))
        # print(target.shape, target.dtype)
        output, h_state = model(data, None)  # get output for every net
        h_state = h_state.data  # repack the hidden state, break the connection from last iteration

        optimizer.zero_grad()  # clear gradients for next train
        loss = loss_func(output, target)  # compute loss for every net
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        loss_total += loss.data.item()
        cnt += data.size(0)

    return loss_total / cnt


def test(model, test_loader):
    model.eval()
    real_target = None
    pred_target = None
    with torch.no_grad():
        for (data, target) in test_loader:
            target = target.data.numpy().reshape(-1, 1)
            real_target = target if real_target is None else np.concatenate((real_target, target))

            output, h_state = model(data, None)  # get output for every net
            h_state = h_state.data  # repack the hidden state, break the connection from last iteration

            y = output.data.numpy()
            pred_target = y if pred_target is None else np.concatenate((pred_target, y))
    return real_target, pred_target


def cal_correction_rate(real_label_array, pred_label_array, num_highest_label, ifprint=True):
    coord = np.vstack((real_label_array, pred_label_array)).T
    coord_argsort = np.argsort(coord, axis=0)
    arg_high_sol = coord_argsort[-num_highest_label:]
    arg_real_high_sol = arg_high_sol[:, 0]
    arg_pred_high_sol = arg_high_sol[:, 1]

    match_coord = np.intersect1d(arg_real_high_sol, arg_pred_high_sol)
    correct_rate = 100. * len(match_coord) / len(arg_high_sol)
    if ifprint:
        print('The correct prediction rate is {:.2f}%'.format(correct_rate))
    return correct_rate


def preproc(df, featurizer_onehot):
    df['mols'] = df.apply(lambda x: Chem.MolFromSmiles(x.smiles), axis=1)
    print(len(df))

    onehot_ID_list = []
    onehot_error_ID_list = []
    onehot_feature_list = []
    onehot_feature_tensor_list = []
    onehot_smiles_list = []
    onehot_label_list = []
    for index, smile, label, mol in zip(list(df.index), list(df['smiles']), list(df['Dielectric Constant']),
                                        list(df['mols'])):
        try:
            feature = featurizer_onehot.featurize([mol])[0]
            # feature_flatten = feature.flatten()
            onehot_ID_list.append(index)
            onehot_feature_list.append(feature)
            onehot_feature_tensor_list.append(torch.tensor(feature))
            onehot_smiles_list.append(smile)
            onehot_label_list.append(label)
        except:
            onehot_error_ID_list.append(index)

    onehot_ID_feature_df = pd.DataFrame([], index=onehot_ID_list)
    onehot_ID_feature_df['ESOL'] = onehot_label_list
    onehot_ID_feature_df.index.name = 'compound_ID'
    print('---Finish OneHotFeaturizer---')

    onehot_label_np = pd.Series.as_matrix(onehot_ID_feature_df['ESOL'])
    onehot_feature_torch = torch.from_numpy(np.asarray(onehot_feature_list, dtype=np.float32))
    onehot_feature_label_torch = torch.from_numpy(onehot_label_np).float()
    return onehot_feature_torch, onehot_feature_label_torch


def preproc_from_vae(csv_file, featurizer_onehot):
    new_df = pd.read_csv(csv_file)
    new_df['mols'] = new_df.apply(lambda x: Chem.MolFromSmiles(x.smiles), axis=1)
    print(len(new_df))

    onehot_ID_list = []
    onehot_feature_list = []
    onehot_feature_tensor_list = []
    onehot_smiles_list = []
    for index, smile, mol in zip(list(new_df.index), list(new_df['smiles']), list(new_df['mols'])):
        feature = featurizer_onehot.featurize([mol])[0]
        onehot_ID_list.append(index)
        onehot_feature_list.append(feature)
        onehot_feature_tensor_list.append(torch.tensor(feature))
        onehot_smiles_list.append(smile)

    # onehot_ID_feature_df = pd.DataFrame([], index=onehot_ID_list)
    print('---Finish OneHotFeaturizer---')

    onehot_feature_torch = torch.from_numpy(np.asarray(onehot_feature_list, dtype=np.float32))
    return onehot_feature_torch


def main(feature_torch, feature_label_torch, network, BATCH_SIZE=16, LR=1e-4, EPOCH=50, n=(35, 64, 1, 120),
         token_order=None):
    CHK_PATH = 'model_RNNdim_' + str(n[1])
    os.makedirs(CHK_PATH)
    torch_dataset = TensorDataset(feature_torch, feature_label_torch)

    train_size = int(0.8 * len(torch_dataset))
    test_size = len(torch_dataset) - train_size
    train_dset, test_dset = random_split(torch_dataset, [train_size, test_size])

    train_loader = DataLoader(
        dataset=train_dset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        # num_workers=2,
    )
    test_loader = DataLoader(
        dataset=test_dset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    print('=== Run RNN problem ===')
    net_regression = network(n[0], n[1], n[2], n[3], 1)
    optim_regression = optim.Adam(net_regression.parameters(), lr=LR)
    loss_regression = nn.MSELoss()

    loss_all = []
    for epoch in range(EPOCH):
        # print('Epoch:', epoch)
        loss = train(net_regression, train_loader, optim_regression, loss_regression, epoch)
        print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, loss))
        loss_all.append(loss)
        if (epoch % 100 == 99) or (epoch == EPOCH - 1):
            test_real_target, test_pred_target = test(net_regression, test_loader)
            if epoch > 1000: torch.save(net_regression.state_dict(), os.path.join(CHK_PATH, str(epoch + 1) + '.pt'))

            plt.scatter(test_real_target, test_pred_target)
            plt.xlabel("Experimental dielectric constant")
            plt.ylabel("Predicted dielectric constant")
            plt.show()

            real_target_flat = test_real_target.flatten()
            pred_target_flat = test_pred_target.flatten()

            # Stat
            pearson = pearsonr(real_target_flat, pred_target_flat)
            mae = mean_absolute_error(real_target_flat, pred_target_flat)
            r2 = r2_score(real_target_flat, pred_target_flat)
            print("pearson correlation is {}, mean absolute error is {}, r2 is {}".format(pearson, mae, r2))

    np.save('RNN_loss_{}'.format(EPOCH), np.array(loss_all))

    plt.plot(loss_all)
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.show()

    # Correction rate of first n large ESOL data
    corr_rate_50 = cal_correction_rate(real_target_flat, pred_target_flat, 50)
    plt.clf()
    plt.xlabel("First n large data")
    plt.ylabel("Percentage (%)")

    # Draw the correction rate from first 1st to first 100th data
    for i in range(1, 101):
        corr_rate = cal_correction_rate(real_target_flat, pred_target_flat, i, ifprint=False)
        plt.scatter(i, corr_rate, color='blue')
    plt.show()
    return net_regression
