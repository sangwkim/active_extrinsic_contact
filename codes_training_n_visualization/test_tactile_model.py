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
import matplotlib.pyplot as plt

###############################################################################

def validation(model, device, test_loader, mode):

    cnn_encoder, decoder = model
    cnn_encoder.eval()
    decoder.eval()

    test_loss = 0

    yy = []
    oo = []

    with torch.no_grad():

        if mode == 'fc':

            for batch_idx, (X11, X12, X21, X22, Y) in enumerate(test_loader):

                X11, X12, X21, X22, Y = X11.to(device), X12.to(device), X21.to(
                    device), X22.to(device), Y.to(device)

                output = decoder(cnn_encoder(X11), cnn_encoder(X12),
                                 cnn_encoder(X21), cnn_encoder(X22), device)

                loss = loss_func(output, Y)

                test_loss += loss.item() * X11.size(0)

                yy.append(Y.cpu().numpy())
                oo.append(output.cpu().numpy())

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

                yy.append(Y.cpu().numpy())
                oo.append(output.cpu().numpy())

    test_loss /= len(test_loader.dataset)

    print('\nTest set : Average loss: {:.6f}\n'.format(test_loss))

    yy = np.vstack(yy)
    oo = np.vstack(oo)

    return test_loss, yy, oo

def test_sequence(model, device, test_set, idx, mode):
    
    cnn_encoder, decoder = model
    cnn_encoder.eval()
    decoder.eval()
    
    if mode == 'fc':

        X11, X12, X21, X22, Y = test_set.load_sequence(test_set.file_folder[idx])
        
        X11, X12, X21, X22, Y = X11.unsqueeze(1).to(device), X12.unsqueeze(1).to(device), X21.unsqueeze(1).to(device), X22.unsqueeze(1).to(device), Y.to(device)
    
        output = decoder(cnn_encoder(X11), cnn_encoder(X12), cnn_encoder(X21), cnn_encoder(X22), device)
        
    elif mode == 'lstm':
        
        X1, X2, Y = test_set.__getitem__(idx)
        
        X1, X2, Y = X1.unsqueeze(0).to(device), X2.unsqueeze(0).to(device), Y.to(device)

        CN1 = cnn_encoder(X1)
        CN2 = cnn_encoder(X2)

        CN1_init = CN1[:, 0, :].repeat([CN1.shape[1], 1,
                                        1]).transpose(0, 1)
        CN2_init = CN2[:, 0, :].repeat([CN2.shape[1], 1,
                                        1]).transpose(0, 1)

        output = decoder(CN1_init, CN1, CN2_init, CN2, device)
    
    return output.squeeze().detach().cpu().numpy(), Y.detach().cpu().numpy()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='fc')
    args = parser.parse_args()

    mode = vars(args)['mode']

    #dataset_name = 'FG_000_210717_large'
    dataset_name = 'FG_000_210731_very_small'
    #xyzypr_limit = np.array(
    #    [1., 2., 2., 2. / 180 * np.pi, 2. / 180 * np.pi, 4. / 180 * np.pi])
    xyzypr_limit = np.array([.25, .5, .5, .2 / 180 * np.pi, .6 / 180 * np.pi,
                    1.2 / 180 * np.pi])
    save_name = "tactile_model"
    obj = 'all'
    obj = dataset_name + '_' + obj
    if mode == 'fc':
        model_name = 'devel_cnn_fc_no_crop'
    elif mode == 'lstm':
        model_name = 'devel_cnn_lstm'
    save_model_path = "../weights/" + model_name + "/" + obj + '/'

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

    cnn_encoder.load_state_dict(torch.load(save_model_path+save_name+'_cnn_encoder_best.pth'))
    decoder.load_state_dict(torch.load(save_model_path+save_name+'_decoder_best.pth'))

    # Parallelize model to multiple GPUs
    if torch.cuda.device_count() > 0:
        print("Using", torch.cuda.device_count(), "GPUs!")
        if torch.cuda.device_count() > 1:
            cnn_encoder = nn.DataParallel(cnn_encoder)
            decoder = nn.DataParallel(decoder)

    model_list = [cnn_encoder, decoder]
    test_loss, yy, oo = validation(model_list, device, valid_loader, mode)
    
    print("test loss: {}".format(test_loss))
    print(yy.shape)
    print(oo.shape)

    for i in range(6):
        plt.figure()
        plt.scatter(yy[:,:,i], oo[:,:,i], s=0.4, alpha=0.5)
        plt.plot([-1,1],[-1,1])
        plt.text(0.5, 0, f'{np.corrcoef(np.reshape(yy[:,:,i],-1),np.reshape(oo[:,:,i],-1))[1,0]:.2f}')
        plt.axis('equal')
        plt.grid()
        plt.show()

    for _ in range(10):
    
        idx = np.random.randint(len(valid_set.file_folder))
    
        pred, gt = test_sequence(model_list, device, valid_set, idx, mode)
        
        plt.figure()
        for i in [1,2,5]:
            plt.plot(pred[:,i], c='C0'+str(i))
        for i in [1,2,5]:
            plt.plot(gt[:,i], c='C0'+str(i), linestyle='--')
        plt.legend(['y','z','roll'])
        plt.xlabel('timestep')
        plt.ylabel('disp (normalized)')
        
    for i in range(6):
        print(np.std(yy[:,:,i]-oo[:,:,i]))
        rs = 1 - np.std(np.clip(yy[:,:,i],-1,1)-oo[:,:,i])**2/np.std(yy[:,:,i])**2
        print(rs)
