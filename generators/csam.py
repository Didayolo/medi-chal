"""SAM: Structural Agnostic Model, Categorical version.

Author: Diviyan Kalainathan
"""
import math
import warnings
import torch as th
from torch.autograd import Variable
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
import pandas as pd
import numpy as np


class CNormalized_Linear(th.nn.Module):
    """Linear layer with column-wise normalized input matrix."""

    def __init__(self, in_features, out_features, bias=False):
        """Initialize the layer."""
        super(CNormalized_Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = th.nn.Parameter(th.Tensor(out_features, in_features))
        if bias:
            self.bias = th.nn.Parameter(th.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters."""
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        """Feed-forward through the network."""
        return th.nn.functional.linear(input, self.weight.div(self.weight.pow(2).sum(0).sqrt()))

    def __repr__(self):
        """For print purposes."""
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) + ')'


class SAM_discriminator(th.nn.Module):
    """Discriminator for the SAM model."""

    def __init__(self, sizes, zero_components=[], **kwargs):
        """Init the SAM discriminator."""
        super(SAM_discriminator, self).__init__()
        self.sht = kwargs.get('shortcut', False)
        activation_function = kwargs.get('activation_function', th.nn.ReLU)
        activation_argument = kwargs.get('activation_argument', None)
        batch_norm = kwargs.get("batch_norm", False)

        layers = []

        for i, j in zip(sizes[:-2], sizes[1:-1]):
            layers.append(th.nn.Linear(i, j))
            if batch_norm:
                layers.append(th.nn.BatchNorm1d(j))
            if activation_argument is None:
                layers.append(activation_function())
            else:
                layers.append(activation_function(activation_argument))

        layers.append(th.nn.Linear(sizes[-2], sizes[-1]))
        self.layers = th.nn.Sequential(*layers)
        # print(self.layers)

    def forward(self, x):
        """Feed-forward the model."""
        return self.layers(x)


class CFilter(th.nn.Module):
    def __init__(self, cat_embedding, zero_components, **kwargs):
        super(CFilter, self).__init__()
        gpu = kwargs.get('gpu', False)
        gpu_no = kwargs.get('gpu_no', 0)

        self._filter = th.ones(1, len(cat_embedding) + 1)
        self.cat = [v for c in [cat_embedding, [1]] for v in c]

        for i in zero_components:
            self._filter[:, i].zero_()

        self.hard_filter = th.cat([f.repeat(i) for f, i
                                   in zip(th.unbind(self._filter, 1),
                                          self.cat)], 0).unsqueeze(0)
        self.hard_filter = Variable(self.hard_filter, requires_grad=False)
        if gpu:
            self.hard_filter = self.hard_filter.cuda(gpu_no)
        self._filter = th.nn.Parameter(self._filter)

    def forward(self, x):
        cfilter = th.cat([f.repeat(i) for f, i
                          in zip(th.unbind(self._filter, 1),
                                 self.cat)], 0).unsqueeze(0)
        # print(x.shape, cfilter.shape)
        return x * (self.hard_filter * cfilter).expand_as(x)


class SAM_block(th.nn.Module):
    """SAM-Block: conditional generator.
    Generates one variable while selecting the parents. Uses filters to do so.
    One fixed filter and one with parameters on order to keep a fixed skeleton.
    """

    def __init__(self, sizes, cat_embedding, zero_components, **kwargs):
        """Initialize a generator."""
        super(SAM_block, self).__init__()

        activation_function = kwargs.get('activation_function', th.nn.Tanh)
        activation_argument = kwargs.get('activation_argument', None)
        batch_norm = kwargs.get("batch_norm", False)
        layers = []
        self.filter = CFilter(cat_embedding, zero_components, **kwargs)

        for i, j in zip(sizes[:-2], sizes[1:-1]):
            layers.append(CNormalized_Linear(i, j))
            if batch_norm:
                layers.append(th.nn.BatchNorm1d(j))
            if activation_argument is None:
                layers.append(activation_function())
            else:
                layers.append(activation_function(activation_argument))

        layers.append(th.nn.Linear(sizes[-2], sizes[-1]))
        self.layers = th.nn.Sequential(*layers)
        self.cat = sizes[-1] != 1
        self.softmax = th.nn.Softmax()

    def forward(self, x):
        """Feed-forward the model."""
        if self.cat:
            return self.softmax(self.layers(self.filter(x)))
        else:
            return self.layers(self.filter(x))


class SAM_generators(th.nn.Module):
    """Ensemble of all the generators."""

    def __init__(self, data_shape, cat_embedding, zero_components, nh=None, batch_size=-1, **kwargs):
        """Init the model."""
        super(SAM_generators, self).__init__()
        if batch_size == -1:
            batch_size = data_shape[0]
        gpu = kwargs.get('gpu', False)
        gpu_no = kwargs.get('gpu_no', 0)
        rows, self.cols = data_shape

        # building the computation graph
        self.noise = [Variable(th.FloatTensor(batch_size, 1))
                      for i in range(self.cols)]
        if gpu:
            self.noise = [i.cuda(gpu_no) for i in self.noise]
        self.blocks = th.nn.ModuleList()

        # Init all the blocks
        for i in range(self.cols):
            self.blocks.append(SAM_block(
                [sum(cat_embedding) + 1, nh, cat_embedding[i]], cat_embedding, zero_components[i], **kwargs))

    def forward(self, x):
        """Feed-forward the model."""
        for i in self.noise:
            i.data.normal_()

        self.generated_variables = [self.blocks[i](
            th.cat([x, self.noise[i]], 1)) for i in range(self.cols)]
        return self.generated_variables


class SAM(object):
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
        super(SAM, self).__init__()
        self.lr = lr
        self.dlr = dlr
        self.l1 = l1
        self.nh = nh
        self.dnh = dnh
        self.train = train_epochs
        self.test = test_epochs
        self.batchsize = batchsize

    def run_SAM(self, df_data, skeleton=None, **kwargs):
        """Execute the SAM model.
        :param df_data:
        """
        gpu = kwargs.get('gpu', False)
        gpu_no = kwargs.get('gpu_no', 0)
        categorical_variables = kwargs.get('categorical_variables', None)

        verbose = kwargs.get('verbose', True)
        plot = kwargs.get("plot", False)
        plot_generated_pair = kwargs.get("plot_generated_pair", False)

        d_str = "Epoch: {} -- Disc: {} -- Gen: {} -- L1: {}"

        if categorical_variables is None:
            warnings.warn("Dataset considered as numerical")
            categorical_variables = [False for i in range(len(df_data.columns))]
        # list_nodes = list(df_data.columns)
        onehotdata = []
        for i, var_is_categorical in enumerate(categorical_variables):
            if var_is_categorical:
                onehotdata.append(pd.get_dummies(df_data.iloc[:, i]).as_matrix())
            else:
                onehotdata.append(df_data.iloc[:, [i]].as_matrix())

        cat_size = [i.shape[1] for i in onehotdata]
        # cat_size.append(1)  # Noise

        df_data = np.concatenate(onehotdata, 1)

        data = df_data.astype('float32')
        data = th.from_numpy(data)
        if self.batchsize == -1:
            self.batchsize = data.shape[0]
        rows, cols = data.size()
        # CAT data: cols override
        cols = len(cat_size)
        # Get the list of indexes to ignore
        if skeleton is not None:
            zero_components = [[] for i in range(cols)]
            for i, j in zip(*((1-skeleton).nonzero())):
                zero_components[j].append(i)
        else:
            zero_components = [[i] for i in range(cols)]
        self.sam = SAM_generators((rows, cols), cat_size, zero_components,
                                  batch_norm=True, nh=self.nh, batch_size=self.batchsize, **kwargs)

        # Begin UGLY
        activation_function = kwargs.get('activation_function', th.nn.Tanh)
        try:
            del kwargs["activation_function"]
        except KeyError:
            pass
        self.discriminator_sam = SAM_discriminator(
            [sum(cat_size), self.dnh, self.dnh, 1], batch_norm=True,
            activation_function=th.nn.LeakyReLU,
            activation_argument=0.2, **kwargs)
        kwargs["activation_function"] = activation_function
        # End of UGLY

        if gpu:
            self.sam = self.sam.cuda(gpu_no)
            self.discriminator_sam = self.discriminator_sam.cuda(gpu_no)
            data = data.cuda(gpu_no)

        # Select parameters to optimize : ignore the non connected nodes
        criterion = th.nn.BCEWithLogitsLoss()
        g_optimizer = th.optim.Adam(self.sam.parameters(), lr=self.lr)
        d_optimizer = th.optim.Adam(
            self.discriminator_sam.parameters(), lr=self.dlr)

        true_variable = Variable(
            th.ones(self.batchsize, 1), requires_grad=False)
        false_variable = Variable(
            th.zeros(self.batchsize, 1), requires_grad=False)
        causal_filters = th.zeros(cols, cols)

        if gpu:
            true_variable = true_variable.cuda(gpu_no)
            false_variable = false_variable.cuda(gpu_no)
            causal_filters = causal_filters.cuda(gpu_no)

        data_iterator = DataLoader(data, batch_size=self.batchsize, shuffle=True, drop_last=True)

        # TRAIN
        for epoch in range(self.train + self.test):
            for i_batch, batch in enumerate(data_iterator):
                batch = Variable(batch)
                # print(batch.size())
                unbind_vectors = th.unbind(batch, 1)
                # print(cat_size)
                batch_vectors = [th.stack(unbind_vectors[sum(cat_size[:idx]):sum(cat_size[:idx]) + i], 1)
                                 if i > 1 else unbind_vectors[sum(cat_size[:idx])].unsqueeze(1)
                                 for idx, i in enumerate(cat_size)]
                g_optimizer.zero_grad()
                d_optimizer.zero_grad()

                # Train the discriminator
                generated_variables = self.sam(batch)
                # for i in generated_variables:
                #     print(i.size())
                # print(batch.size())
                disc_losses = []
                gen_losses = []
                # print([j.size() for j in batch_vectors])
                # print([j.size() for j in generated_variables])

                for i in range(cols):
                    generator_output = th.cat([v for c in [batch_vectors[: i], [
                        generated_variables[i]],
                        batch_vectors[i + 1:]] for v in c], 1)

                    # 1. Train discriminator on fake
                    # print(i, generator_output.size())
                    disc_output_detached = self.discriminator_sam(
                        generator_output.detach())
                    disc_output = self.discriminator_sam(generator_output)
                    disc_losses.append(
                        criterion(disc_output_detached, false_variable))

                    # 2. Train the generator :
                    gen_losses.append(criterion(disc_output, true_variable))

                true_output = self.discriminator_sam(batch)
                adv_loss = sum(disc_losses)/cols + \
                    criterion(true_output, true_variable)
                gen_loss = sum(gen_losses)

                adv_loss.backward()
                d_optimizer.step()

                # 3. Compute filter regularization
                filters = th.stack(
                    [i.filter._filter[0, :-1].abs() for i in self.sam.blocks], 1)
                l1_reg = self.l1 * filters.sum()
                loss = gen_loss + l1_reg

                if verbose:

                    print(str(i) + " " + d_str.format(epoch,
                                                      adv_loss.cpu().data[0],
                                                      gen_loss.cpu(
                                                      ).data[0] / cols,
                                                      l1_reg.cpu().data[0]))
                loss.backward()
                # STORE ASSYMETRY values for output
                if epoch >= self.train:
                    causal_filters.add_(filters.data)
                g_optimizer.step()

                if plot and i_batch == 0:
                    try:
                        ax.clear()
                        ax.plot(range(len(adv_plt)), adv_plt, "r-",
                                linewidth=1.5, markersize=4,
                                label="Discriminator")
                        ax.plot(range(len(adv_plt)), gen_plt, "g-", linewidth=1.5,
                                markersize=4, label="Generators")
                        ax.plot(range(len(adv_plt)), l1_plt, "b-",
                                linewidth=1.5, markersize=4,
                                label="L1-Regularization")
                        ax.plot(range(len(adv_plt)), asym_plt, "c-",
                                linewidth=1.5, markersize=4,
                                label="Assym penalization")

                        plt.legend()

                        adv_plt.append(adv_loss.cpu().data[0])
                        gen_plt.append(gen_loss.cpu().data[0] / cols)
                        l1_plt.append(l1_reg.cpu().data[0])
                        asym_plt.append(asymmetry_reg.cpu().data[0])
                        plt.pause(0.0001)

                    except NameError:
                        plt.ion()
                        plt.figure()
                        plt.xlabel("Epoch")
                        plt.ylabel("Losses")

                        plt.pause(0.0001)

                        adv_plt = [adv_loss.cpu().data[0]]
                        gen_plt = [gen_loss.cpu().data[0] / cols]
                        l1_plt = [l1_reg.cpu().data[0]]

                elif plot:
                    adv_plt.append(adv_loss.cpu().data[0])
                    gen_plt.append(gen_loss.cpu().data[0] / cols)
                    l1_plt.append(l1_reg.cpu().data[0])

                if plot_generated_pair and i_batch == 0:
                    if epoch == 0:
                        plt.ion()
                        to_print = [[0, 1]]  # , [1, 0]]  # [2, 3]]  # , [11, 17]]
                        plt.clf()
                    for (i, j) in to_print:

                        plt.scatter(generated_variables[i].data.cpu().numpy(
                        ), batch.data.cpu().numpy()[:, j], label="Y -> X")
                        plt.scatter(batch.data.cpu().numpy()[
                            :, i], generated_variables[j].data.cpu().numpy(), label="X -> Y")

                        plt.scatter(batch.data.cpu().numpy()[:, i], batch.data.cpu().numpy()[
                            :, j], label="original data")
                        plt.legend()

                    plt.pause(0.01)

        return causal_filters.div_(self.test).cpu().numpy()

    def predict(self, data, categorical_variables=None, skeleton=None, nruns=1, njobs=1, gpus=0, verbose=True,
                plot=False, plot_generated_pair=False):
        """Execute SAM on a dataset given a skeleton or not.
        :param data: Observational data for estimation of causal relationships by SAM
        :param skeleton: A priori knowledge about the causal relationships as an adjacency matrix.
                         Can be fed either directed or undirected links.
        :param nruns: Number of runs to be made for causal estimation.
                      Recommended: >=12 for optimal performance.
        :param njobs: Numbers of jobs to be run in Parallel.
                      Recommended: 1 if no GPU available, 2*number of GPUs else.
        :param gpus: Number of available GPUs for the algorithm.
        :param verbose: verbose mode
        :param plot: Plot losses interactively. Not recommended if nruns>1
        :param plot_generated_pair: plots a generated pair interactively.  Not recommended if nruns>1
        :return: Adjacency matrix (A) of the graph estimated by SAM,
                A[i,j] is the term of the ith variable for the jth generator.
        """
        list_out = []
        for i in range(nruns):
            list_out.append(self.run_SAM(data,
                                         categorical_variables=categorical_variables,
                                         skeleton=skeleton,                            gpu=bool(gpus),
                                         plot=plot, verbose=verbose, gpu_no=0))

        W = list_out[0]
        for w in list_out[1:]:
            W += w
        W /= nruns
        return W

    def generate(self, df_data, categorical_variables=None, **kwargs):
        gpu = kwargs.get('gpu', False)
        gpu_no = kwargs.get('gpu_no', 0)
        if categorical_variables is None:
            warnings.warn("Dataset considered as numerical")
            categorical_variables = [False for i in range(len(df_data.columns))]
        # list_nodes = list(df_data.columns)
        onehotdata = []
        for i, var_is_categorical in enumerate(categorical_variables):
            if var_is_categorical:
                onehotdata.append(pd.get_dummies(df_data.iloc[:, i]).as_matrix())
            else:
                onehotdata.append(df_data.iloc[:, [i]].as_matrix())

        cat_size = [i.shape[1] for i in onehotdata]
        # cat_size.append(1)  # Noise

        df_data = np.concatenate(onehotdata, 1)

        data = df_data.astype('float32')
        data = th.from_numpy(data)
        if gpu:
            data = data.cuda(gpu_no)
        return self.sam(data)
