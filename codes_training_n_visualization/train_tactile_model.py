import os, shutil
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import numpy as np
import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
from torch.autograd import Variable
from torch.optim import lr_scheduler
from dataset_devel import data_selection
from networkmodels import DecoderFC, DecoderRNN, EncoderCNN
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import gc
gc.collect()
import argparse

###############################################################################


def train(model, device, train_loader, optimizer, epoch, mode):

    cnn_encoder, decoder = model
    cnn_encoder.train()
    decoder.train()
    train_loss = 0

    if mode == 'fc':

        for batch_idx, (X11, X12, X21, X22, Y) in enumerate(train_loader):

            X11, X12, X21, X22, Y = X11.to(device), X12.to(device), X21.to(
                device), X22.to(device), Y.to(device)

            optimizer.zero_grad()

            output = decoder(cnn_encoder(X11), cnn_encoder(X12),
                             cnn_encoder(X21), cnn_encoder(X22), device)

            loss = loss_func(output, Y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X11.size(0)

    elif mode == 'lstm':

        for batch_idx, (X1, X2, Y) in enumerate(train_loader):

            X1, X2, Y = X1.to(device), X2.to(device), Y.to(device)

            optimizer.zero_grad()

            CN1 = cnn_encoder(X1)
            CN2 = cnn_encoder(X2)

            CN1_init = CN1[:, 0, :].repeat([CN1.shape[1], 1,
                                            1]).transpose(0, 1)
            CN2_init = CN2[:, 0, :].repeat([CN2.shape[1], 1,
                                            1]).transpose(0, 1)

            output = decoder(CN1_init, CN1, CN2_init, CN2, device)

            loss = loss_func(output, Y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X1.size(0)

    train_loss /= len(train_loader.dataset)

    print('\nTrain set : Average loss: {:.6f}\n'.format(train_loss))

    return train_loss


def validation(model, device, optimizer, test_loader, epoch, min_loss, mode):

    cnn_encoder, decoder = model
    cnn_encoder.eval()
    decoder.eval()

    test_loss = 0

    with torch.no_grad():

        if mode == 'fc':

            for batch_idx, (X11, X12, X21, X22, Y) in enumerate(test_loader):

                X11, X12, X21, X22, Y = X11.to(device), X12.to(device), X21.to(
                    device), X22.to(device), Y.to(device)

                output = decoder(cnn_encoder(X11), cnn_encoder(X12),
                                 cnn_encoder(X21), cnn_encoder(X22), device)

                loss = loss_func(output, Y)

                test_loss += loss.item() * X11.size(0)

        elif mode == 'lstm':

            for batch_idx, (X1, X2, Y) in enumerate(test_loader):

                X1, X2, Y = X1.to(device), X2.to(device), Y.to(device)

                CN1 = cnn_encoder(X1)
                CN2 = cnn_encoder(X2)

                CN1_init = CN1[:, 0, :].repeat([CN1.shape[1], 1,
                                                1]).transpose(0, 1)
                CN2_init = CN2[:, 0, :].repeat([CN2.shape[1], 1,
                                                1]).transpose(0, 1)

                output = decoder(CN1_init, CN1, CN2_init, CN2, device)

                loss = loss_func(output, Y)

                test_loss += loss.item() * X1.size(0)

    test_loss /= len(test_loader.dataset)

    print('\nTest set : Average loss: {:.6f}\n'.format(test_loss))

    if test_loss < min_loss:
        min_loss = test_loss
        torch.save(
            cnn_encoder.state_dict(),
            os.path.join(save_model_path, save_name +
                         '_cnn_encoder_best.pth'), _use_new_zipfile_serialization=False)  # save spatial_encoder
        torch.save(decoder.state_dict(),
                   os.path.join(save_model_path, save_name +
                                '_decoder_best.pth'), _use_new_zipfile_serialization=False)  # save motion_encoder

        print("Epoch {} model saved!".format(epoch + 1))

    return test_loss, min_loss


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='fc')
    args = parser.parse_args()

    mode = vars(args)['mode']

    dataset_name = 'FG_000_210731_very_small' #'FG_000_210717_large'
    xyzypr_limit = np.array([.25, .5, .5, .2 / 180 * np.pi, .6 / 180 * np.pi,
                    1.2 / 180 * np.pi])
    #np.array(
    #    [1., 2., 2., 2. / 180 * np.pi, 2. / 180 * np.pi, 4. / 180 * np.pi])

    save_name = "tactile_model"
    obj = 'all'
    obj = dataset_name + '_' + obj
    if mode == 'fc':
        model_name = f'devel_cnn_fc_no_crop'
    elif mode == 'lstm':
        model_name = f'devel_cnn_lstm'
    save_model_path = "../weights/" + model_name + "/" + obj + '/'

    load_weight = False
    epochs = 50

    learning_rate = 1e-4

    loss_func = torch.nn.MSELoss()

    if mode == 'fc':
        pairs_for_seq = 10
        sequence_len = None
    elif mode == 'lstm':
        pairs_for_seq = None
        sequence_len = 70

    # EncoderCNN architecture
    CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
    in_channels = 3
    CNN_embed_dim = 512  # latent dim extracted by 2D CNN
    img_x, img_y = 300, 218
    dropout_cnn = 0.  # dropout probability

    if mode == 'fc':
        # DecoderFC architecture
        FC_layer_nodes = [512, 512, 256]
        dropout_fc = 0.15
        k = 6  # output_dims (XYZYPR)
    elif mode == 'lstm':
        # DecoderRNN architecture
        RNN_hidden_layers = 2
        RNN_hidden_nodes = 512
        RNN_FC_dim = 256
        dropout_rnn = 0.5
        k = 6  # output_dims (XYZYPR)
    ###############################################################################

    if not os.path.exists("../weights/" + model_name):
        os.mkdir("../weights/" + model_name)
    if not os.path.exists("../results/" + model_name):
        os.mkdir("../results/" + model_name)
    if not os.path.exists("../results/" + model_name + '/' + obj):
        os.mkdir("../results/" + model_name + '/' + obj)
        shutil.copy(
            __file__,
            "../results/" + model_name + '/' + obj + '/main_autosaved.py')
        shutil.copy(
            'networkmodels.py',
            "../results/" + model_name + '/' + obj + '/network_autosaved.py')
        shutil.copy(
            'dataset_devel.py',
            "../results/" + model_name + '/' + obj + '/utils_autosaved.py')
    if not os.path.exists(save_model_path):
        os.mkdir(save_model_path)

    if mode == 'fc':
        batch_size = 128
    elif mode == 'lstm':
        batch_size = 12
    train_set, valid_set, train_data_size, valid_data_size = data_selection(
        dataset_name, xyzypr_limit, sequence_len, pairs_for_seq, mode)
    print('training data size: ', train_data_size, 'validation data size: ',
          valid_data_size)
    params = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': 4,
        'pin_memory': True
    }
    train_loader = torch.utils.data.DataLoader(train_set, **params)
    valid_loader = torch.utils.data.DataLoader(valid_set, **params)

    use_cuda = torch.cuda.is_available()  # check if GPU exists
    device = torch.device("cuda:0" if use_cuda else "cpu")  # use CPU or GPU

    # Create model
    cnn_encoder = EncoderCNN(img_x=img_x,
                             img_y=img_y,
                             input_channels=in_channels,
                             fc_hidden1=CNN_fc_hidden1,
                             fc_hidden2=CNN_fc_hidden2,
                             drop_p=dropout_cnn,
                             CNN_embed_dim=CNN_embed_dim).to(device)

    if mode == 'fc':
        decoder = DecoderFC(CNN_embed_dim=CNN_embed_dim,
                            FC_layer_nodes=FC_layer_nodes,
                            drop_p=dropout_fc,
                            output_dim=k).to(device)
    elif mode == 'lstm':
        decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim,
                             h_RNN_layers=RNN_hidden_layers,
                             h_RNN=RNN_hidden_nodes,
                             h_FC_dim=RNN_FC_dim,
                             drop_p=dropout_rnn,
                             output_dim=k).to(device)

    if load_weight:
        print("weight loading function is not developed yet")
    else:
        epoch_train_losses = []
        epoch_valid_losses = []

    # Parallelize model to multiple GPUs
    if torch.cuda.device_count() > 0:
        print("Using", torch.cuda.device_count(), "GPUs!")
        if torch.cuda.device_count() > 1:
            cnn_encoder = nn.DataParallel(cnn_encoder)
            decoder = nn.DataParallel(decoder)

    params = list(cnn_encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    # record training process

    # start training
    min_loss = 1000.
    for epoch in range(epochs):
        # train, test model
        print('epoch:', epoch)
        for param_group in optimizer.param_groups:
            print('lr', param_group['lr'])
        model_list = [cnn_encoder, decoder]
        train_losses = train(model_list, device, train_loader, optimizer,
                             epoch, mode)
        test_loss, min_loss = validation(model_list, device, optimizer,
                                         valid_loader, epoch, min_loss, mode)
        scheduler.step()

        # save results
        epoch_train_losses.append(train_losses)
        epoch_valid_losses.append(test_loss)

        np.save(
            '../results/' + model_name + '/' + obj + '/training_losses.npy',
            np.array(epoch_train_losses))
        np.save(
            '../results/' + model_name + '/' + obj + '/validation_losses.npy',
            np.array(epoch_valid_losses))