import argparse
import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tensorflow.contrib.layers import l2_regularizer
from tensorflow.contrib.layers import batch_norm

_VALIDATION_RATIO = 0.1


class Medgan(object):
    def __init__(self,
                 dataType='binary',
                 inputDim=615,
                 embeddingDim=128,
                 randomDim=128,
                 generatorDims=(128, 128),
                 discriminatorDims=(256, 128, 1),
                 compressDims=(),
                 decompressDims=(),
                 bnDecay=0.99,
                 l2scale=0.001):
        self.inputDim = inputDim
        self.embeddingDim = embeddingDim
        self.generatorDims = list(generatorDims) + [embeddingDim]
        self.randomDim = randomDim
        self.dataType = dataType

        if dataType == 'binary':
            self.aeActivation = tf.nn.sigmoid
        else:
            self.aeActivation = tf.nn.relu

        self.generatorActivation = tf.nn.relu
        self.discriminatorActivation = tf.nn.relu
        self.discriminatorDims = discriminatorDims
        self.compressDims = list(compressDims) + [embeddingDim]
        self.decompressDims = list(decompressDims) + [inputDim]
        self.bnDecay = bnDecay
        self.l2scale = l2scale

    def loadData(self, dataPath=''):
        full_data = np.load(dataPath)

        if self.dataType == 'binary':
            full_data = np.clip(full_data, 0, 1)

        trainX, validX = train_test_split(
            full_data, test_size=_VALIDATION_RATIO, random_state=0)

        # save for future testing
        np.save('data/trainX.npy', trainX)
        np.save('data/validX.npy', validX)

        return trainX, validX

    def buildAutoencoder(self, x_input):
        decodeVariables = {}
        with tf.variable_scope('autoencoder',
                               regularizer=l2_regularizer(self.l2scale)):
            tempVec = x_input
            tempDim = self.inputDim
            i = 0
            # default is self.compressDims = [128]
            for compressDim in self.compressDims:
                W = tf.get_variable('aee_W_' + str(i),
                                    shape=[tempDim, compressDim])
                b = tf.get_variable('aee_b_' + str(i), shape=[compressDim])
                tempVec = self.aeActivation(tf.add(tf.matmul(tempVec, W), b))
                tempDim = compressDim
                i += 1

            # implement sparse autoencoder
            def kl_divergence(rho, rho_hat):
                return rho * tf.log(rho) - rho * tf.log(rho_hat + 1e-12) + \
                       (1 - rho) * tf.log(1 - rho) - \
                       (1 - rho) * tf.log(1 - rho_hat + 1e-12)

            # sparsity parameter
            rho = 0.01
            # calculate kl loss portion
            rho_hat = tf.reduce_mean(tempVec, axis=0)
            kl = kl_divergence(rho, rho_hat)
            beta = 3

            i = 0
            # default is self.decompressDims[:-1] = []
            for decompressDim in self.decompressDims[:-1]:
                W = tf.get_variable('aed_W_' + str(i),
                                    shape=[tempDim, decompressDim])
                b = tf.get_variable('aed_b_' + str(i), shape=[decompressDim])
                tempVec = self.aeActivation(tf.add(tf.matmul(tempVec, W), b))
                tempDim = decompressDim
                decodeVariables['aed_W_' + str(i)] = W
                decodeVariables['aed_b_' + str(i)] = b
                i += 1

            W = tf.get_variable('aed_W_' + str(i),
                                shape=[tempDim, self.decompressDims[-1]])
            b = tf.get_variable('aed_b_' + str(i),
                                shape=[self.decompressDims[-1]])

            decodeVariables['aed_W_' + str(i)] = W
            decodeVariables['aed_b_' + str(i)] = b

            if self.dataType == 'binary':
                x_reconst = tf.nn.sigmoid(tf.add(tf.matmul(tempVec, W), b))
                loss = tf.reduce_mean(-tf.reduce_sum(x_input * tf.log(
                    x_reconst + 1e-12) + (1. - x_input) * tf.log(
                    1. - x_reconst + 1e-12), 1), 0) + beta * tf.reduce_sum(kl)
            else:
                x_reconst = tf.nn.relu(tf.add(tf.matmul(tempVec, W), b))
                loss = tf.reduce_mean((x_input - x_reconst)**2)

        return loss, decodeVariables

    def buildGenerator(self, x_input, bn_train):
        tempVec = x_input
        tempDim = self.randomDim
        with tf.variable_scope('generator',
                               regularizer=l2_regularizer(self.l2scale)):
            i = 0
            for i, genDim in enumerate(self.generatorDims[:-1]):
                W = tf.get_variable('W_' + str(i), shape=[tempDim, genDim])
                h = tf.matmul(tempVec, W)
                h2 = batch_norm(h, decay=self.bnDecay, scale=True,
                                is_training=bn_train, updates_collections=None)
                h3 = self.generatorActivation(h2)
                tempVec = h3 + tempVec
                tempDim = genDim
            W = tf.get_variable(
                'W_' + str(i + 1), shape=[tempDim, self.generatorDims[-1]])
            h = tf.matmul(tempVec, W)
            h2 = batch_norm(h, decay=self.bnDecay, scale=True,
                            is_training=bn_train, updates_collections=None)

            if self.dataType == 'binary':
                h3 = tf.nn.tanh(h2)
            else:
                h3 = tf.nn.relu(h2)

            output = h3 + tempVec
        return output

    def buildGeneratorTest(self, x_input, bn_train):
        tempVec = x_input
        tempDim = self.randomDim
        with tf.variable_scope('generator',
                               regularizer=l2_regularizer(self.l2scale)):
            i = 0
            for i, genDim in enumerate(self.generatorDims[:-1]):
                W = tf.get_variable('W_' + str(i), shape=[tempDim, genDim])
                h = tf.matmul(tempVec, W)
                # only difference between this and previous is
                # the "trainable=False" param
                h2 = batch_norm(h, decay=self.bnDecay, scale=True,
                                is_training=bn_train, updates_collections=None,
                                trainable=False)
                h3 = self.generatorActivation(h2)
                tempVec = h3 + tempVec
                tempDim = genDim
            W = tf.get_variable(
                'W_' + str(i + 1), shape=[tempDim, self.generatorDims[-1]])
            h = tf.matmul(tempVec, W)
            # only difference between this and previous is
            # the "trainable=False" param
            h2 = batch_norm(h, decay=self.bnDecay, scale=True,
                            is_training=bn_train, updates_collections=None,
                            trainable=False)

            if self.dataType == 'binary':
                h3 = tf.nn.tanh(h2)
            else:
                h3 = tf.nn.relu(h2)

            output = h3 + tempVec
        return output

    def getDiscriminatorResults(self, x_input, keepRate, reuse=False):
        batchSize = tf.shape(x_input)[0]
        colSize = tf.shape(x_input)[1]
        inputMeanCol = tf.reshape(tf.tile(tf.reduce_mean(x_input, 0),
                                          [batchSize]),
                                  (batchSize, self.inputDim))
        #inputMeanRow = tf.reshape(tf.tile(tf.reduce_mean(x_input, 1),
        #                                  [colSize]),
        #                          (batchSize, self.inputDim))
        #tempVec = tf.concat([x_input, inputMeanCol, inputMeanRow], 1)
        tempVec = tf.concat([x_input, inputMeanCol], 1)
        tempDim = self.inputDim * 2
        with tf.variable_scope('discriminator', reuse=reuse,
                               regularizer=l2_regularizer(self.l2scale)):
            for i, discDim in enumerate(self.discriminatorDims[:-1]):
                W = tf.get_variable('W_' + str(i), shape=[tempDim, discDim])
                b = tf.get_variable('b_' + str(i), shape=[discDim])
                h = self.discriminatorActivation(
                    tf.add(tf.matmul(tempVec, W), b))
                h = tf.nn.dropout(h, keepRate)
                tempVec = h
                tempDim = discDim
            W = tf.get_variable('W', shape=[tempDim, 1])
            b = tf.get_variable('b', shape=[1])
            y_hat = tf.squeeze(tf.nn.sigmoid(tf.add(tf.matmul(tempVec, W), b)))
        return y_hat

    def buildDiscriminator(self, x_real, x_fake, keepRate,
                           decodeVariables):
        # Discriminate for real samples
        y_hat_real = self.getDiscriminatorResults(
            x_real, keepRate, reuse=False)

        # Decompress, then discriminate for real samples
        tempVec = x_fake
        i = 0
        for _ in self.decompressDims[:-1]:
            tempVec = self.aeActivation(tf.add(tf.matmul(
                tempVec, decodeVariables['aed_W_' + str(i)]),
                decodeVariables['aed_b_' + str(i)]))
            i += 1

        if self.dataType == 'binary':
            x_decoded = tf.nn.sigmoid(tf.add(tf.matmul(
                tempVec, decodeVariables['aed_W_' + str(i)]),
                decodeVariables['aed_b_' + str(i)]))
        else:
            x_decoded = tf.nn.relu(tf.add(tf.matmul(
                tempVec, decodeVariables['aed_W_' + str(i)]),
                decodeVariables['aed_b_' + str(i)]))

        y_hat_fake = self.getDiscriminatorResults(
            x_decoded, keepRate, reuse=True)

        loss_d = -tf.reduce_mean(tf.log(y_hat_real + 1e-12)) - \
            tf.reduce_mean(tf.log(1. - y_hat_fake + 1e-12))
        loss_g = -tf.reduce_mean(tf.log(y_hat_fake + 1e-12))

        return loss_d, loss_g, y_hat_real, y_hat_fake

    @staticmethod
    def print2file(buf, outFile):
        outfd = open(outFile, 'a')
        outfd.write(buf + '\n')
        outfd.close()

    def generateData(self,
                     nSamples=100,
                     modelFile='model',
                     batchSize=100,
                     outFile='out'):
        x_dummy = tf.placeholder('float', [None, self.inputDim])
        _, decodeVariables = self.buildAutoencoder(x_dummy)
        x_random = tf.placeholder('float', [None, self.randomDim])
        bn_train = tf.placeholder('bool')
        x_emb = self.buildGeneratorTest(x_random, bn_train)
        tempVec = x_emb
        i = 0
        for _ in self.decompressDims[:-1]:
            tempVec = self.aeActivation(tf.add(tf.matmul(
                tempVec, decodeVariables['aed_W_' + str(i)]),
                decodeVariables['aed_b_' + str(i)]))
            i += 1

        if self.dataType == 'binary':
            x_reconst = tf.nn.sigmoid(tf.add(tf.matmul(
                tempVec, decodeVariables['aed_W_' + str(i)]),
                decodeVariables['aed_b_' + str(i)]))
        else:
            x_reconst = tf.nn.relu(tf.add(tf.matmul(
                tempVec, decodeVariables['aed_W_' + str(i)]),
                decodeVariables['aed_b_' + str(i)]))

        np.random.seed(1234)
        saver = tf.train.Saver()
        outputVec = []
        burn_in = 1000
        with tf.Session() as sess:
            saver.restore(sess, modelFile)
            print('burning in')
            for i in range(burn_in):
                randomX = np.random.normal(size=(batchSize, self.randomDim))
                _ = sess.run(x_reconst, feed_dict={
                                  x_random: randomX, bn_train: True})

            print('generating')
            nBatches = int(np.ceil(float(nSamples)) / float(batchSize))
            for i in range(nBatches):
                randomX = np.random.normal(size=(batchSize, self.randomDim))
                output = sess.run(x_reconst, feed_dict={
                                  x_random: randomX, bn_train: False})
                outputVec.extend(output)

        outputMat = np.array(outputVec)
        np.save(outFile, outputMat)

    @staticmethod
    def calculateDiscAuc(preds_real, preds_fake):
        preds = np.concatenate([preds_real, preds_fake], axis=0)
        labels = np.concatenate(
            [np.ones((len(preds_real))), np.zeros((len(preds_fake)))], axis=0)
        auc = roc_auc_score(labels, preds)
        return auc

    @staticmethod
    def calculateDiscAccuracy(preds_real, preds_fake):
        total = len(preds_real) + len(preds_fake)
        hit = 0
        for pred in preds_real:
            if pred > 0.5:
                hit += 1
        for pred in preds_fake:
            if pred < 0.5:
                hit += 1
        acc = float(hit) / float(total)
        return acc

    def train(self,
              dataPath='data',
              modelPath='',
              outPath='out',
              nEpochs=500,
              discriminatorTrainPeriod=2,
              generatorTrainPeriod=1,
              pretrainBatchSize=100,
              batchSize=1000,
              pretrainEpochs=100,
              saveMaxKeep=0):
        x_raw = tf.placeholder('float', [None, self.inputDim])
        x_random = tf.placeholder('float', [None, self.randomDim])
        keep_prob = tf.placeholder('float')
        bn_train = tf.placeholder('bool')

        # build sub scopes of graph
        loss_ae, decodeVariables = self.buildAutoencoder(x_raw)
        x_fake = self.buildGenerator(x_random, bn_train)
        loss_d, loss_g, y_hat_real, y_hat_fake = self.buildDiscriminator(
            x_raw, x_fake, keep_prob, decodeVariables)
        trainX, validX = self.loadData(dataPath)

        # get variables and separate by scope
        t_vars = tf.trainable_variables()
        ae_vars = [var for var in t_vars if 'autoencoder' in var.name]
        d_vars = [var for var in t_vars if 'discriminator' in var.name]
        g_vars = [var for var in t_vars if 'generator' in var.name]

        all_regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        # create optimizers
        optimize_ae = tf.train.AdamOptimizer().minimize(
            loss_ae + sum(all_regs), var_list=ae_vars)
        optimize_d = tf.train.AdamOptimizer().minimize(
            loss_d + sum(all_regs), var_list=d_vars)
        optimize_g = tf.train.AdamOptimizer().minimize(
            loss_g + sum(all_regs),
            var_list=g_vars + list(decodeVariables.values()))

        initOp = tf.global_variables_initializer()

        nBatches = int(np.ceil(float(trainX.shape[0]) / float(batchSize)))
        saver = tf.train.Saver(max_to_keep=saveMaxKeep)
        logFile = outPath + '.log'

        tf.summary.FileWriter('medgan_tensorboard', graph=tf.get_default_graph())

        with tf.Session() as sess:
            if modelPath == '':
                sess.run(initOp)
            else:
                saver.restore(sess, modelPath)
            nTrainBatches = int(
                np.ceil(float(trainX.shape[0])) / float(pretrainBatchSize))
            nValidBatches = int(
                np.ceil(float(validX.shape[0])) / float(pretrainBatchSize))

            if modelPath == '':
                for epoch in range(pretrainEpochs):
                    idx = np.random.permutation(trainX.shape[0])
                    trainLossVec = []
                    for i in range(nTrainBatches):
                        batchX = trainX[idx[i * pretrainBatchSize:
                                            (i + 1) * pretrainBatchSize]]
                        _, loss = sess.run(
                            [optimize_ae, loss_ae], feed_dict={x_raw: batchX})
                        trainLossVec.append(loss)
                    idx = np.random.permutation(validX.shape[0])
                    validLossVec = []
                    for i in range(nValidBatches):
                        batchX = validX[idx[i * pretrainBatchSize:
                                            (i + 1) * pretrainBatchSize]]
                        loss = sess.run(loss_ae, feed_dict={x_raw: batchX})
                        validLossVec.append(loss)
                    validReverseLoss = 0.
                    buf = 'Pretrain_Epoch:{}, trainLoss:{}, validLoss:{}, ' \
                          'validReverseLoss:{}'.format(epoch,
                                                       np.mean(trainLossVec),
                                                       np.mean(validLossVec),
                                                       validReverseLoss)
                    print(buf)
                    self.print2file(buf, logFile)

            idx = np.arange(trainX.shape[0])
            savePath = ''
            for epoch in range(nEpochs):
                d_loss_vec = []
                g_loss_vec = []
                for i in range(nBatches):
                    batchX = trainX
                    for _ in range(discriminatorTrainPeriod):
                        batchIdx = np.random.choice(
                            idx, size=batchSize, replace=False)
                        batchX = trainX[batchIdx]
                        randomX = np.random.normal(
                            size=(batchSize, self.randomDim))
                        _, discLoss = sess.run([optimize_d, loss_d],
                                               feed_dict={
                                                   x_raw: batchX,
                                                   x_random: randomX,
                                                   keep_prob: 1.0,
                                                   bn_train: False})
                        d_loss_vec.append(discLoss)
                    for _ in range(generatorTrainPeriod):
                        randomX = np.random.normal(
                            size=(batchSize, self.randomDim))
                        _, generatorLoss = sess.run([optimize_g, loss_g],
                                                    feed_dict={
                                                        x_raw: batchX,
                                                        x_random: randomX,
                                                        keep_prob: 1.0,
                                                        bn_train: True})
                        g_loss_vec.append(generatorLoss)

                idx = np.arange(validX.shape[0])
                validAccVec = []
                validAucVec = []
                for i in range(nBatches):
                    batchIdx = np.random.choice(
                        idx, size=batchSize, replace=False)
                    batchX = validX[batchIdx]
                    randomX = np.random.normal(
                        size=(batchSize, self.randomDim))
                    preds_real, preds_fake, = sess.run([y_hat_real,
                                                        y_hat_fake],
                                                       feed_dict={
                                                           x_raw: batchX,
                                                           x_random: randomX,
                                                           keep_prob: 1.0,
                                                           bn_train: False})

                    validAcc = self.calculateDiscAccuracy(
                        preds_real, preds_fake)
                    validAuc = self.calculateDiscAuc(preds_real, preds_fake)
                    validAccVec.append(validAcc)
                    validAucVec.append(validAuc)
                buf = 'Epoch:{}, d_loss:{}, g_loss:{}, accuracy:{}, ' \
                      'AUC:{}'.format(epoch, np.mean(d_loss_vec),
                                      np.mean(g_loss_vec),
                                      np.mean(validAccVec),
                                      np.mean(validAucVec))
                print(buf)
                self.print2file(buf, logFile)
                savePath = saver.save(sess, os.path.join(
                    os.getcwd(), outPath), global_step=epoch)
        print(savePath)

    def train_autoencoder(self,
                          dataPath='data',
                          outPath='out',
                          pretrainBatchSize=100,
                          pretrainEpochs=100):
        """just train the autoencoder"""
        x_raw = tf.placeholder('float', [None, self.inputDim])

        # build sub scopes of graph
        loss_ae, decodeVariables = self.buildAutoencoder(x_raw)
        trainX, validX = self.loadData(dataPath)

        # get variables and separate by scope
        t_vars = tf.trainable_variables()
        ae_vars = [var for var in t_vars if 'autoencoder' in var.name]

        all_regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        # create optimizers
        optimize_ae = tf.train.AdamOptimizer().minimize(
            loss_ae + sum(all_regs), var_list=ae_vars)

        initOp = tf.global_variables_initializer()

        logFile = outPath + '.log'

        tf.summary.scalar("loss", loss_ae)
        merged_summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter('medgan_tensorboard',
                                               graph=tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(initOp)
            nTrainBatches = int(
                np.ceil(float(trainX.shape[0])) / float(pretrainBatchSize))
            nValidBatches = int(
                np.ceil(float(validX.shape[0])) / float(pretrainBatchSize))

            for epoch in range(pretrainEpochs):
                idx = np.random.permutation(trainX.shape[0])
                trainLossVec = []
                for i in range(nTrainBatches):
                    batchX = trainX[idx[i * pretrainBatchSize:
                                        (i + 1) * pretrainBatchSize]]
                    _, loss, summary = sess.run(
                        [optimize_ae, loss_ae, merged_summary_op],
                        feed_dict={x_raw: batchX})

                    summary_writer.add_summary(summary,
                                               epoch * nTrainBatches + i)
                    trainLossVec.append(loss)
                idx = np.random.permutation(validX.shape[0])
                validLossVec = []
                for i in range(nValidBatches):
                    batchX = validX[idx[i * pretrainBatchSize:
                                        (i + 1) * pretrainBatchSize]]
                    loss = sess.run(loss_ae, feed_dict={x_raw: batchX})
                    validLossVec.append(loss)
                validReverseLoss = 0.
                buf = 'Pretrain_Epoch:{}, trainLoss:{}, validLoss:{}, ' \
                      'validReverseLoss:{}'.format(epoch,
                                                   np.mean(trainLossVec),
                                                   np.mean(validLossVec),
                                                   validReverseLoss)
                print(buf)
                self.print2file(buf, logFile)

        summary_writer.close()


def parse_arguments(parser):
    parser.add_argument('--embed_size', type=int, default=128,
                        help='The dimension size of the embedding, which '
                             'will be generated by the generator. (default '
                             'value: 128)')
    parser.add_argument('--noise_size', type=int, default=128,
                        help='The dimension size of the random noise, '
                             'on which the generator is conditioned. ('
                             'default value: 128)')
    parser.add_argument('--generator_size', type=tuple, default=(128, 128),
                        help='The dimension size of the generator. Note that '
                             'another layer of size "--embed_size" is always '
                             'added. (default value: (128, 128))')
    parser.add_argument('--discriminator_size', type=tuple,
                        default=(256, 128, 1),
                        help='The dimension size of the discriminator. ('
                             'default value: (256, 128, 1))')
    parser.add_argument('--compressor_size', type=tuple, default=(),
                        help='The dimension size of the encoder of the '
                             'autoencoder. Note that another layer of size '
                             '"--embed_size" is always added. Therefore this '
                             'can be a blank tuple. (default value: ())')
    parser.add_argument('--decompressor_size', type=tuple, default=(),
                        help='The dimension size of the decoder of the '
                             'autoencoder. Note that another layer, '
                             'whose size is equal to the dimension of the '
                             '<patient_matrix>, is always added. Therefore '
                             'this can be a blank tuple. (default value: ())')
    parser.add_argument('--data_type', type=str, default='binary', choices=[
                        'binary', 'count'],
                        help='The input data type. The <patient matrix> '
                             'could either contain binary values or count '
                             'values. (default value: "binary")')
    parser.add_argument('--batchnorm_decay', type=float, default=0.99,
                        help='Decay value for the moving average used in '
                             'Batch Normalization. (default value: 0.99)')
    parser.add_argument('--L2', type=float, default=0.001,
                        help='L2 regularization coefficient for all weights. '
                             '(default value: 0.001)')

    parser.add_argument('data_file', type=str, metavar='<patient_matrix>',
                        help='The path to the numpy matrix containing '
                             'aggregated patient records.')
    parser.add_argument('out_file', type=str, metavar='<out_file>',
                        help='The path to the output models.')
    parser.add_argument('--model_file', type=str, metavar='<model_file>',
                        default='',
                        help='The path to the model file, in case you want '
                             'to continue training. (default value: '')')
    parser.add_argument('--n_pretrain_epoch', type=int, default=100,
                        help='The number of epochs to pre-train the '
                             'autoencoder. (default value: 100)')
    parser.add_argument('--n_epoch', type=int, default=1000,
                        help='The number of epochs to train medGAN. (default '
                             'value: 1000)')
    parser.add_argument('--n_discriminator_update', type=int, default=2,
                        help='The number of times to update the '
                             'discriminator per epoch. (default value: 2)')
    parser.add_argument('--n_generator_update', type=int, default=1,
                        help='The number of times to update the generator '
                             'per epoch. (default value: 1)')
    parser.add_argument('--pretrain_batch_size', type=int, default=100,
                        help='The size of a single mini-batch for '
                             'pre-training the autoencoder. (default value: '
                             '100)')
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='The size of a single mini-batch for training '
                             'medGAN. (default value: 1000)')
    parser.add_argument('--save_max_keep', type=int, default=0,
                        help='The number of models to keep. Setting this to '
                             '0 will save models for every epoch. (default '
                             'value: 0)')
    parser.add_argument('--generate', action='store_true',
                        help="Activate generate mode")
    parser.add_argument('--autoencoder', action='store_true',
                        help="Activate autoencoder mode")
    parsed_args = parser.parse_args()
    return parsed_args


if __name__ == '__main__':
    args = parse_arguments(argparse.ArgumentParser())

    data = np.load(args.data_file)

    mg = Medgan(dataType=args.data_type,
                inputDim=data.shape[1],
                embeddingDim=args.embed_size,
                randomDim=args.noise_size,
                generatorDims=args.generator_size,
                discriminatorDims=args.discriminator_size,
                compressDims=args.compressor_size,
                decompressDims=args.decompressor_size,
                bnDecay=args.batchnorm_decay,
                l2scale=args.L2)

    if args.autoencoder:
        mg.train_autoencoder(dataPath=args.data_file,
                             outPath=args.out_file,
                             pretrainBatchSize=100,
                             pretrainEpochs=100)

    elif not args.generate:
        mg.train(dataPath=args.data_file,
                 modelPath=args.model_file,
                 outPath=args.out_file,
                 pretrainEpochs=args.n_pretrain_epoch,
                 nEpochs=args.n_epoch,
                 discriminatorTrainPeriod=args.n_discriminator_update,
                 generatorTrainPeriod=args.n_generator_update,
                 pretrainBatchSize=args.pretrain_batch_size,
                 batchSize=args.batch_size,
                 saveMaxKeep=args.save_max_keep)

    # To generate synthetic data using a trained model:
    # Comment the train function above and un-comment generateData function
    # below.
    # You must specify "--model_file" and "<out_file>" to generate synthetic
    # data.
    else:
        mg.generateData(nSamples=10000,
                        modelFile=args.model_file,
                        batchSize=args.batch_size,
                        outFile=args.out_file)
