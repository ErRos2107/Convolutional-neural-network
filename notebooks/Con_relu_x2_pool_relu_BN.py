
import matplotlib.pyplot as plt

plt.style.use('ggplot')
import numpy as np
from mlp.layers import AffineLayer, SoftmaxLayer, SigmoidLayer, ReluLayer, LeakyReluLayer, ELULayer, SELULayer
from mlp.layers import ConvolutionalLayer, MaxPoolingLayer, ReshapeLayer, BatchNormalizationLayer
from mlp.errors import CrossEntropySoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import ConstantInit, GlorotUniformInit
from mlp.learning_rules import GradientDescentLearningRule
from mlp.learning_rules import RMSPropLearningRule, AdamLearningRule
from mlp.optimisers import Optimiser
from mlp.penalty import L1Penalty, L2Penalty
from collections import defaultdict
import logging
import pickle
from mlp.data_providers import MNISTDataProvider, EMNISTDataProvider

def train_model_and_plot_stats(
        model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=False):

    # As well as monitoring the error over training also monitor classification
    # accuracy i.e. proportion of most-probable predicted classes being equal to targets
    data_monitors={'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}

    # Use the created objects to initialise a new Optimiser instance.
    optimiser = Optimiser(
        model, error, learning_rule, train_data, valid_data, data_monitors, notebook=notebook)

    # Run the optimiser for 5 epochs (full passes through the training set)
    # printing statistics every epoch.
    stats, keys, run_time = optimiser.train(num_epochs=num_epochs, stats_interval=stats_interval)

    # Plot the change in the validation and training set error over training.
    fig_1 = plt.figure(figsize=(8, 4))
    ax_1 = fig_1.add_subplot(111)
    for k in ['error(train)', 'error(valid)']:
        ax_1.plot(np.arange(1, stats.shape[0]) * stats_interval,
                  stats[1:, keys[k]], label=k)
    ax_1.legend(loc=0)
    ax_1.set_xlabel('Epoch number')

    # Plot the change in the validation and training set accuracy over training.
    fig_2 = plt.figure(figsize=(8, 4))
    ax_2 = fig_2.add_subplot(111)
    for k in ['acc(train)', 'acc(valid)']:
        ax_2.plot(np.arange(1, stats.shape[0]) * stats_interval,
                  stats[1:, keys[k]], label=k)
    ax_2.legend(loc=0)
    ax_2.set_xlabel('Epoch number')

    return optimiser, stats, keys, run_time, fig_1, ax_1, fig_2, ax_2

################################################################################
# save and present the data
def save_and_present(experiment, stats, parameter):

    # save the model to disk
    filename = experiment +'.sav'
    pickle.dump(model, open(filename, 'wb'))

    np.savetxt(experiment+'_'+str(parameter)+'.csv', stats, delimiter=',')

    error_valid= stats[1:, keys['error(valid)']]
    error_train= stats[1:, keys['error(train)']]
    acc_valid = stats[1:, keys['acc(valid)']]

    file = open(experiment+'_'+str(parameter)+'.txt','w')

    overfitting = error_valid-error_train
    file.write('Experiment '+experiment+' best acc at Epoch={} by parameter={}\n'.
          format(np.argmax(acc_valid)+1, parameter))
    file.write('error(train)= {}, error(valid)={}, \n error gap = {},  acc(valid)={}\n'.
          format(error_train[np.argmax(acc_valid)],error_valid[np.argmax(acc_valid)],overfitting[np.argmax(acc_valid)], max(acc_valid)))
    file.write('Smallest error gap(after best acc epoch) = {} at Epoch={}'.
          format(min(overfitting[np.argmax(acc_valid):]),np.argmin(overfitting[np.argmax(acc_valid):])+np.argmax(acc_valid)+1))
    print('Experiment '+experiment+' best acc at Epoch={} by parameter={}\n'.
          format(np.argmax(acc_valid)+1, parameter))
    print('error(train)= {}, error(valid)={}, \n error gap = {},  acc(valid)={}\n'.
          format(error_train[np.argmax(acc_valid)],error_valid[np.argmax(acc_valid)],overfitting[np.argmax(acc_valid)], max(acc_valid)))
    print('Smallest error gap(after best acc epoch) = {} at Epoch={}'.
          format(min(overfitting[np.argmax(acc_valid):]),np.argmin(overfitting[np.argmax(acc_valid):])+np.argmax(acc_valid)+1))

# Seed a random number generator
seed = 10102016
rng = np.random.RandomState(seed)
batch_size = 100
# Set up a logger object to print info about the training run to stdout
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = [logging.StreamHandler()]

# Create data provider objects for the EMNIST data set
train_data = EMNISTDataProvider('train', batch_size=batch_size, rng=rng)
valid_data = EMNISTDataProvider('valid', batch_size=batch_size, rng=rng)

####################################################################################################################################################
# to ensure reproducibility of results
rng.seed(seed)

#setup hyperparameters
learning_rate = 1e-3
num_epochs = 50
stats_interval = 1

pad=0
stride=1
# kernel shape and feature maps
num_output_channels1, num_output_channels2, kernel_dim_1, kernel_dim_2 = 5,10,5,5
# Initial input, final output shape
inputs_units, output_dim = 784, 47
#####################################################################################################
# Rehape to image shape for first convol
num_input_channels, input_dim_1, input_dim_2 = 1, 28, 28
# the ouput shape of the first convol layer is (batch_size, num_output_channels, Con_out_1, Con_out_1)
Con_out_1 =  (input_dim_1 - kernel_dim_1+2*pad)//stride + 1
# Flatten the image for relu

#####################################################################################################
# The input shape of the second conv layer

# the ouput shape of the second convol layer is (batch_size, num_output_channels2, Con_out_2, Con_out_2)
Con_out_2 = (Con_out_1 - kernel_dim_1+2*pad)//stride + 1
# Flatten the image for relu
# Rehape to image shape for maxpool

# the ouput shape of the Maxpool
Max_out = Con_out_2//2
#####################################################################################################
# then flatten the output
hidden_dim = num_output_channels2* Max_out* Max_out

weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)

model = MultipleLayerModel([
    ReshapeLayer((num_input_channels,input_dim_1,input_dim_2)),

    ConvolutionalLayer(num_input_channels, num_output_channels1, input_dim_1, input_dim_2, kernel_dim_1, kernel_dim_2),
    ReluLayer(),

    ConvolutionalLayer(num_output_channels1, num_output_channels2, Con_out_1, Con_out_1, kernel_dim_1, kernel_dim_2),
    ReluLayer(),

	MaxPoolingLayer(),

	ReshapeLayer(),
    ReluLayer(),
	BatchNormalizationLayer(hidden_dim),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

error = CrossEntropySoftmaxError()
# learning rule
learning_rule = AdamLearningRule(learning_rate=learning_rate,)

experiment = 'Con_relu_x2_pool_relu_BN'

#return stats, keys, run_time, fig_1, ax_1, fig_2, ax_2
optimiser, stats, keys, run_time, fig_1, ax_1, fig_2, ax_2 = train_model_and_plot_stats(
    model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)
fig_1.savefig(experiment+ '_learning_rate_{}_error.pdf'.format(learning_rate))
fig_2.savefig(experiment+ '_learning_rate_{}_accuracy.pdf'.format(learning_rate))

save_and_present(experiment, stats, learning_rate)

result = optimiser.eval_monitors(test_data, 'test')
print('Test error:    ' + str(save_results[experiment]['errortest']))
print('Test accuracy: ' + str(save_results[experiment]['acctest']))
