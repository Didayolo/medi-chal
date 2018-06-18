"""Structural Agnostic Model.

Author: Diviyan Kalainathan, Olivier Goudet
Date: 09/3/2018
"""
import math
import numpy as np
import torch as th
import pandas as pd
from time import time, sleep
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from sklearn.preprocessing import scale
from utils.linear3d import Linear3D
from utils.batchnorm import ChannelBatchNorm1d
from multiprocessing import Pool
from torch.nn import functional as f
# from .gsam_v0 import SAM_generators 


class SAM_generators(th.nn.Module):
    """Ensemble of all the generators."""

    def __init__(self, data_shape, nh, cat_sizes, mask):
        """Init the model."""
        super(SAM_generators, self).__init__()
        layers = []
        self.input_layer = Linear3D(len(cat_sizes), data_shape[1], nh, noise=True, 
                                    batch_size=data_shape[0], normalize=True)
        layers.append(ChannelBatchNorm1d(len(cat_sizes), nh))
        layers.append(th.nn.Tanh())
        layers.append(Linear3D(len(cat_sizes), nh, max(cat_sizes)))
        # self.weights = Linear3D(data_shape[1], data_shape[1], 1)
        self.layers = th.nn.Sequential(*layers) 
        self.o_sizes = cat_sizes
        self.adjacency_matrix = th.nn.Parameter(th.ones(len(cat_sizes), len(cat_sizes)))
        self.register_buffer('mask', mask)
            
    def forward(self, data):
        """Forward through all the generators."""
        # print(data.shape, adj_matrix.shape, drawn_neurons.shape, self.input_layer(data, adj_matrix).shape)
        output = self.layers(self.input_layer(data, self.mask.t() @ self.adjacency_matrix))
        return th.cat([f.softmax(output[:, idx, :i], dim=1) if i>0 else th.output[:, idx, :i]
                       for idx, i in enumerate(self.o_sizes)], 1)
    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        
      
class SAM_discriminator(th.nn.Module):
    """SAM discriminator."""

    def __init__(self, nfeatures, dnh, mask=None):
        super(SAM_discriminator, self).__init__()
        self.nfeatures = nfeatures
        layers = []
        layers.append(th.nn.Linear(nfeatures, dnh))
        layers.append(th.nn.BatchNorm1d(dnh))
        layers.append(th.nn.LeakyReLU(.2))
        layers.append(th.nn.Linear(dnh, dnh))
        layers.append(th.nn.BatchNorm1d(dnh))
        layers.append(th.nn.LeakyReLU(.2))
        layers.append(th.nn.Linear(dnh, 1))
        self.layers = th.nn.Sequential(*layers)

        if mask is None:
            mask = th.eye(n_features, n_features)
        self.register_buffer("mask", mask.unsqueeze(0))

    def forward(self, input, obs_data=None):
        if obs_data is not None:
            # print((obs_data.unsqueeze(1) * (1-self.mask)
            #                     + sum([input.unsqueezei) for i in (1) * self.mask).shape)
            return sum([self.layers(i) for i in th.unbind(obs_data.unsqueeze(1) * (1-self.mask)
                                                          + input.unsqueeze(1) * self.mask, 1)])
            # return self.layers((obs_data.unsqueeze(1) * (1-self.mask)
            #                     + input.unsqueeze(1) * self.mask).view(-1, obs_data.shape[1]))
            
            
        else:
            return self.layers(input)
    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
  
def run_SAM(in_data, categorical_variables, skeleton=None, device="cpu", train=1000, test=1000,
            batch_size=-1, lr_gen=.01, lr_disc=.01, regul_param=.1, nh=None, dnh=None, drawhard=True, tau=1,
            verbose=True, plot=False, return_model=False):
    """Run SAM on data."""
    d_str = "Epoch: {} -- Disc: {:.4f} --  Total: {:.4f} -- Gen: {:.4f} -- L1: {:.4f}"
    # print("KLPenal:{}, fganLoss:{}".format(KLpenalization, fganLoss))
    list_nodes = list(in_data.columns)
    onehotdata = []
    for i, var_is_categorical in enumerate(categorical_variables):
        if var_is_categorical:
            onehotdata.append(pd.get_dummies(in_data.iloc[:, i]).values)
        else:
            onehotdata.append(scale(in_data.iloc[:, [i]].values))

    cat_size = [i.shape[1] for i in onehotdata]
    features = len(cat_size)
    th_cat_size = th.FloatTensor(cat_size).to(device)
    data = np.concatenate(onehotdata, 1)# data = scale(in_data[list_nodes].values)
    data = data.astype('float32')
    data = th.from_numpy(data).to(device)

    if batch_size == -1:
        batch_size = data.shape[0]
    rows, cols = data.size()
    # print(data.shape)
    # Get the list of indexes to ignore
    if skeleton is not None:
        skeleton = th.from_numpy(skeleton.astype('float32')).to(device)
    else: 
        skeleton = th.ones(cols, cols).to(device)
        for cat, cumul in zip(cat_size, np.cumsum(cat_size)):
            skeleton[cumul-cat:cumul, cumul-cat:cumul].zero_()
    mask = th.zeros(features, sum(cat_size))
    for idx, (cat, cumul) in enumerate(zip(cat_size, np.cumsum(cat_size))):
        mask[idx,cumul-cat:cumul].fill_(1)
    sam = SAM_generators((batch_size, cols), nh, cat_size, mask).to(device)
    discriminator = SAM_discriminator(cols, dnh, mask).to(device)
    sam.reset_parameters()
    discriminator.reset_parameters()
    criterion = th.nn.BCEWithLogitsLoss()
    g_optimizer = th.optim.Adam(sam.parameters(), lr=lr_gen) #, betas=)
    d_optimizer = th.optim.Adam(discriminator.parameters(), lr=lr_disc) # , betas=(0.5, 0.9))

    _true = th.ones(1).to(device)
    _false = th.zeros(1).to(device)
    output = th.zeros(features, features).to(device)

    data_iterator = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)

    # TRAIN
    for epoch in range(train + test):
        for i_batch, batch in enumerate(data_iterator):

            g_optimizer.zero_grad()
            d_optimizer.zero_grad()
            # Train the discriminator
            generated_variables = sam(batch)
            disc_vars_d = discriminator(generated_variables.detach(), batch) / cols
            disc_vars_g = discriminator(generated_variables, batch) / cols
            true_vars_disc = discriminator(batch)
            disc_loss = criterion(disc_vars_d, _false.expand_as(disc_vars_d)) \
                        + criterion(true_vars_disc, _true.expand_as(true_vars_disc))
            # Gen Losses per generator: multiply py the number of channels
            gen_loss = criterion(disc_vars_g, _true.expand_as(disc_vars_g)) * cols

            disc_loss.backward()
            d_optimizer.step()
            filters = sam.adjacency_matrix.abs()
            loss = gen_loss + regul_param * filters.sum()  
            
            if verbose and epoch % 10 == 0 and i_batch == 0:

                print(str(i_batch) + " " + d_str.format(epoch, disc_loss.item(),
                                                        loss.item(),
                                                        gen_loss.item()/ cols,
                                                        filters.sum() 
                                                        ))
            loss.backward()

            # STORE ASSYMETRY values for output
            if epoch >= train:
                output.add_(filters.data)
            # if not epoch % 5:
            g_optimizer.step()

            if plot and epoch % 40 == 0:
                if epoch == 0:
                    plt.ion()
                to_print = [[0, 2]]  # , [1, 0]]  # [2, 3]]  # , [11, 17]]
                plt.clf()
                for (i, j) in to_print:

                    plt.scatter(generated_variables[:, i].data.cpu().numpy(
                    ), batch.data.cpu().numpy()[:, j], label="Y -> X")
                    plt.scatter(batch.data.cpu().numpy()[
                        :, i], generated_variables[:, j].data.cpu().numpy(), label="X -> Y")

                    plt.scatter(batch.data.cpu().numpy()[:, i], batch.data.cpu().numpy()[
                        :, j], label="original data")
                    plt.legend()

                plt.pause(0.01)
    if return_model:
        return output.div_(test).cpu().numpy(), sam, data 
    else:
        return output.div_(test).cpu().numpy()


class catSAM(object):
    """Structural Agnostic Model."""

    def __init__(self, lr=0.1, dlr=0.1, l1=0.1, nh=200, dnh=200,
                 train_epochs=1000, test_epochs=1000, batchsize=-1):
        """Init and parametrize the SAM model.

        :param lr: Learning rate of the generators
        :param dlr: Learning rate of the discriminator
        :param l1: L1 penalization on the causal filters
        :param nh: Number of hidden units in the generators' hidden layers
        :param dnh: Number of hidden units in the discriminator's hidden layer$
        :param train_epochs: Number of training epochs
        :param test_epochs: Number of test epochs (saving and averaging the causal filters)
        :param batchsize: Size of the batches to be fed to the SAM model.
        """
        super(catSAM, self).__init__()
        self.lr = lr
        self.dlr = dlr
        self.l1 = l1
        self.nh = nh
        self.dnh = dnh
        self.train = train_epochs
        self.test = test_epochs
        self.batchsize = batchsize
        
    def exec_sam_instance(self, data, categorical_variables, skeleton=None, gpus=0, 
                          gpuno=0, verbose=True, plot=False, return_model=False):
            # print(seed); sleep(seed)
            device = "cuda:{}".format(gpuno) if bool(gpus) else "cpu"
            return run_SAM(data, categorical_variables, skeleton=skeleton, lr_gen=self.lr, lr_disc=self.dlr,
                           regul_param=self.l1, nh=self.nh, dnh=self.dnh,
                           device=device, train=self.train,
                           test=self.test, batch_size=self.batchsize, return_model=return_model)

    def predict(self, data, categorical_variables, skeleton=None, nruns=6, njobs=1, gpus=0,
                verbose=True, plot=False, return_model=False):
        """Execute SAM on a dataset given a skeleton or not.

        :param data: Observational data for estimation of causal relationships by SAM
        :param skeleton: A priori knowledge about the causal relationships as an adjacency matrix.
                         Can be fed either directed or undirected links.
        :param nruns: Number of runs to be made for causal estimation.
                      Recommended: >5 for optimal performance.
        :param njobs: Numbers of jobs to be run in Parallel.
                      Recommended: 1 if no GPU available, 2*number of GPUs else.
        :param gpus: Number of available GPUs for the algorithm.
        :param verbose: verbose mode
        :param plot: Plot losses interactively. Not recommended if nruns>1
        :param plot_generated_pair: plots a generated pair interactively.  Not recommended if nruns>1
        :return: Adjacency matrix (A) of the graph estimated by SAM,
                A[i,j] is the term of the ith variable for the jth generator.
        """
        assert nruns > 0
        if nruns == 1:
            return self.exec_sam_instance(data, categorical_variables, skeleton=skeleton, plot=plot,
                                          verbose=verbose, gpus=gpus, return_model=return_model)
        else:
            if return_model:
                raise ValueError("Choose nruns=1 to return model")
            list_out = Parallel(n_jobs=njobs)(delayed(self.exec_sam_instance)(
                                              data, categorical_variables, gpus=gpus, skeleton=skeleton,
                                              verbose=verbose, plot=plot,
                                              gpuno=idx % gpus if gpus else 0) 
                                              for idx in range(nruns))

            W = list_out[0]
            for w in list_out[1:]:
                W += w
            W /= nruns
            return W
