"""
this script contains most of the necessary classes and defines test functions used
in the later scripts for 2D function analysis 

this is written predominantly by kerwin and hughes, and was only minimally modified
"""
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import sys
import os
sys.path.append('../')
import two_dimensional_function
import pandas as pd

theta_1_min = 0
theta_1_max = 4
theta_2_min = 0
theta_2_max = 4
f_min = 0
f_max = 1

n_samples = 100

state = two_dimensional_function.state(a=1/np.sqrt(2), b=1/np.sqrt(2))
# define toy models via lambda functions
function = lambda theta_1, theta_2: two_dimensional_function.control_landscape(state, theta_1, theta_2)


class ToyModels:

    def __init__(self, n_samples, theta_1_min, theta_1_max, theta_2_min, theta_2_max, 
                 seed=None):

        """
        :param n_samples: the number of samples
        """

        self.n_samples = n_samples
        self.theta_1_min = theta_1_min
        self.theta_1_max = theta_1_max
        self.theta_2_min = theta_2_min
        self.theta_2_max = theta_2_max
        np.random.seed(seed)

    def create_noise(self, n_samples, f, noise):

        """
        :param theta: domain values
        :param noise: list containing elements [type of noise, param1, param2]
        :return: array of noise values
        """

        if noise[0] == 'g':
            noise_arr = np.random.normal(float(noise[1]), float(noise[2]), size=(n_samples))
            
        elif noise[0] == 'b':
            noise_arr_1 = f - np.random.binomial(float(noise[1]), f, theta_1.shape) / float(noise[1])
            noise_arr_2 = f - np.random.binomial(float(noise[1]), f, theta_2.shape) / float(noise[1])

        elif noise[0] == 'n':
            return np.zeros((n_samples, n_samples))

        else:
            raise Exception(f'Unexpected parameter {noise[0]}.')

        return noise_arr

    def qubit(self, noise):
        
        theta_1_range = self.theta_1_max - self.theta_1_min
        theta_2_range = self.theta_2_max - self.theta_2_min
        
        theta_1 = np.array((theta_1_range * np.random.random(self.n_samples)) + self.theta_1_min)
        theta_2 = np.array((theta_2_range * np.random.random(self.n_samples)) + self.theta_2_min)

        #theta_1 = np.array(sorted(theta_1))
        #theta_2 = np.array(sorted(theta_2))
        
        f_val=[]
        for i in range(len(theta_1)):
            f = function(theta_1[i], theta_2[i])
            Noise = self.create_noise(self.n_samples, f, noise)
            f += Noise[i]
            f_val.append(f)
            
        
        return theta_1.reshape((-1, 1)), theta_2.reshape((-1,1)), f_val

    

# A class to handle normalisation and standardisation of data
class Normaliser:

    def normalise(self, arr):
        """
        :param arr: array to be normalised
        :return: a normalised array
        """

        self.mean = np.mean(arr)
        self.std = np.std(arr)

        return (arr - self.mean) / self.std

    def standardise(self, arr):
        self.mean = np.mean(arr)
        self.range = np.amax(arr) - np.amin(arr)

        return 2 * (arr - self.mean) / self.range
    

    def normalise2(self, arr):
        """
        :param arr: array to be normalised
        :return: a normalised array
        """

        return (arr - self.mean) / self.std

    def standardise2(self, arr):
        return 2 * (arr - self.mean) / self.range

    def denormalise(self, arr):
        return self.mean + (self.std * arr)

    def destandardise(self, arr):
        return self.mean + ((self.range / 2) * arr)

    def denormalise_var(self, arr):
        return (self.std ** 2) * arr

    def destandardise_var(self, arr):
        return ((self.range / 2) ** 2) * arr


# A custom loss function for negative log likelihood, with the assumption that data follows a gaussian random variable.
def custom_loss(sigma):
    """
    :param sigma: the variance at each epoch
    :return: the NLL loss function
    """

    def nll_loss(z_true, z_pred):
        '''
        :param y_true: true z values
        :param y_pred: predicted z values
        :return: loss
        '''
        z_diff = tf.math.subtract(z_true, z_pred)
        loss = tf.math.reduce_mean(.5 * tf.math.log(sigma) + .5 * tf.math.divide(tf.math.square(z_diff), sigma))
        return loss

    return nll_loss


# A custom layer allowing for two outputs for mean and standard deviation
class OutputLayer(tf.keras.layers.Layer):
    """
    A custom output layer which inherits from tf.keras.layers.Layer
    """

    def __init__(self, **kwargs):
        """
        :param num_outputs: the number of outputs of the custom output layer
        :param kwargs: key word argument
        """
        super(OutputLayer, self).__init__(**kwargs)
# super just means same initialisation as the parent class 
    def build(self, input_shape):
        # assuming this is constructing what the layer is
        """
        :param input_shape: shape of the input passed from the previous layer
        """

        # add new weights to the layer
        # the method add_weight is a method of the parent class 
        # the glorot (also called xavier) initialisation helps stop variance blowing up
        self.kernel_mu = self.add_weight(shape=(input_shape[1], 1),
                                         initializer=tf.keras.initializers.glorot_normal(), trainable=True)
        self.kernel_sigma = self.add_weight(shape=(input_shape[1], 1),
                                            initializer=tf.keras.initializers.glorot_normal(), trainable=True)

        # add bias neurons to the layer
        # shape(1,) means a single number 
        self.bias_mu = self.add_weight(shape=(1,),
                                       initializer=tf.keras.initializers.glorot_normal(), trainable=True)
        self.bias_sigma = self.add_weight(shape=(1,),
                                          initializer=tf.keras.initializers.glorot_normal(), trainable=True)

        super(OutputLayer, self).build(input_shape)

    def call(self, x, **kwargs):
        # assuming this defines the computation from inputs to outputs 
        """
        :param x:
        :return: a tensor containing the output of the hidden unit
        """
# here K is the tensorflow keras backend 
        output_mu = K.dot(x, self.kernel_mu) + self.bias_mu
        # this is basically the modified input that goes into the activation function
        
        # output_mu_activated = tf.keras.activations.tanh(output_mu)
        
        output_sigma = K.dot(x, self.kernel_sigma) + self.bias_sigma
        output_sigma_activated = tf.keras.activations.softplus(output_sigma) + 1e-6

        return output_mu, output_sigma_activated


# A custom binomial loss function using negative log likelihood and assuming data follows a binomial random variable.
def binomial_loss(y_true, y_pred):
    '''
    :param y_true: true y data
    :param y_pred: predicted y data from the model at the end of each epoch
    :return: loss value for each epoch
    '''
    n, N = tf.unstack(y_true, num=2, axis=-1)
    p, N_pred = tf.unstack(y_pred, num=2, axis=-1)
    loss = -tf.math.lgamma(N + 1) + tf.math.lgamma(N - n + 1) + tf.math.lgamma(n + 1)
    loss -= tf.math.multiply(n, tf.math.log(p))
    loss -= tf.math.multiply((N - n), tf.math.log(1 - p))
    return tf.math.reduce_mean(loss)


# A function to handle saving errors to a .txt file
def error_handling(file_name):
    '''
    :param file_name: errors associated with the file_name ran
    :return: null --- but writes error to file
    '''
    error_path = os.path.join(os.pardir,'error-logs')
                                  
    try:
        os.mkdir(error_path)
    except FileExistsError as e:
        print(e)
    sys.stderr = open(os.path.join(error_path, f'{file_name}.txt'), 'w')


# A function to make pathways and directories for binomial model.
def model_pathways_make_directories_binomial(toymodel, model_name, n_samples, vary_N, N):
    '''
    :param toymodel: the toy model used (singlequbit or cubic)
    :param model_name: the chosen model name (e.g. Ben)
    :param n_samples: the number of data samples fed into the neural network
    :param vary_N: whether the number of repetitions of measurements at each value of x is repeated
    :param N: the number of repetitions
    :return: the pathways required
    '''
    data_path = os.path.join(os.pardir, "data")
    data_path = os.path.join(data_path, toymodel)
    data_path = os.path.join(data_path, 'binomial')
    data_path = os.path.join(data_path, vary_N)
    data_path = os.path.join(data_path, N)
    results_path = os.path.join(os.pardir, 'results')
    model_path = os.path.join(results_path, f'{model_name}-{n_samples}')
    loss_curves_path = os.path.join(model_path, 'loss-curves')

    make_directories(results_path, model_path, loss_curves_path)

    return data_path, results_path, model_path, loss_curves_path


# A function to make pathways and directories for regular (non-binomial) models.
def model_pathways_make_directories_regular(folder, ensemble_size, toymodel, model_name, n_samples, noise_info):
    '''
    :param toymodel: the toy model used
    :param model_name: the model name
    :param n_samples: the number of samples
    
    folder is the folder you want to save to
    
    :param noise_info: the noise information passed in from command line arguments
    :return: the pathways required
    '''
    if folder is None: 
        data_path = os.path.join(os.pardir, "data")
    else: 
        data_path = os.path.join(folder, "data")
        
 #   print("these are the two things that we want to join", data_path, toymodel)
    # identified that toymodel has to be a string (as there's a corresponding folder for single_qubit in the data folder)
    data_path = os.path.join(data_path, toymodel)
    data_path = os.path.join(data_path, 'regular')

    noise_str = noise(noise_info)

    if noise_info[0] == 'g':
        data_path = os.path.join(data_path, 'gaussian')
    elif noise_info[0] == 'b':
        data_path = os.path.join(data_path, 'binomial')
    else:
        data_path = os.path.join(data_path, 'noiseless')
        
    if folder is None: 
        results_path = os.path.join(os.pardir, "results")
    else: 
        results_path_1 = os.path.join(folder, f"{ensemble_size} models")
        try:
            os.mkdir(results_path_1)
        except FileExistsError as e:
            print(e)
        results_path = os.path.join(results_path_1, "results")
    
    model_path = os.path.join(results_path, f'{model_name}-{n_samples}{noise_str}')

    loss_curves_path = os.path.join(model_path, 'loss-curves')

    make_directories(results_path, model_path, loss_curves_path)

    return data_path, results_path, model_path, loss_curves_path


# Create a noise string
    # this is just a part of a folder name to characterise the noise of the data
def noise(noise_info):
    '''
    :param noise_info: noise information passed in from command line arguments
    :return: a noise string
    '''
    if noise_info[0] == 'n':
        noise_str = ''
    elif noise_info[0] == 'g':
        noise_str = f'-{noise_info[0]}noise-{noise_info[1]}-{noise_info[2]}'
    elif noise_info[0] == 'b':
        noise_str = f'-{noise_info[0]}noise-{noise_info[1]}'
    else:
        raise Exception(f'Unexpected parameter {noise_info[0]}.')

    return noise_str


def make_directories(results_path, model_path, loss_curves_path):
    '''
    :param results_path: the results path
    :param model_path: the model path
    :param loss_curves_path: the loss curves path
    :return: null
    '''
        
    try:
        os.mkdir(results_path)
    except FileExistsError as e:
        print(e)

    try:
        os.mkdir(model_path)
    except FileExistsError as e:
        print(e)

    try:
        os.mkdir(loss_curves_path)
    except FileExistsError as e:
        print(e)


# A function to make data pathways and directories for binomial data.
def data_pathways_make_directories_binomial(toymodel, vary_N, N):
    '''
    :param toymodel: the toy model used
    :param vary_N: whether repetitions are varied or not
    :param N: the number of repetitions
    :return: null
    '''
    data_path = os.path.join(os.pardir, 'data')

    try:
        os.mkdir(data_path)
    except FileExistsError as e:
        print(e)

    data_path = os.path.join(data_path, toymodel)

    try:
        os.mkdir(data_path)
    except FileExistsError as e:
        print(e)

    data_path = os.path.join(data_path, 'binomial')

    try:
        os.mkdir(data_path)
    except FileExistsError as e:
        print(e)

    data_path = os.path.join(data_path, vary_N)

    try:
        os.mkdir(data_path)
    except FileExistsError as e:
        print(e)

    data_path = os.path.join(data_path, str(N))

    try:
        os.mkdir(data_path)
    except FileExistsError as e:
        print(e)

    return data_path


# A function to make data pathways and directories for regular data.
def data_pathways_make_directories_regular(folder, toymodel,noise):
    '''
    :param toymodel: the toy model used
    :param noise: the noise information
    :return: null
    '''

    if folder is None: 
        data_path = os.path.join(os.pardir, "data")
    else: 
        data_path = os.path.join(folder, "data")
        
    try:
        os.mkdir(data_path)
    except FileExistsError as e:
        print(e)

    data_path = os.path.join(data_path, toymodel)

    try:
        os.mkdir(data_path)
    except FileExistsError as e:
        print(e)

    data_path = os.path.join(data_path, 'regular')

    try:
        os.mkdir(data_path)
    except FileExistsError as e:
        print(e)

    if noise[0] == 'g':
        data_path = os.path.join(data_path, 'gaussian')
    elif noise[0] == 'b':
        data_path = os.path.join(data_path, 'binomial')
    else:
        data_path = os.path.join(data_path, 'noiseless')

    try:
        os.mkdir(data_path)
    except FileExistsError as e:
        print(e)

    return data_path


# Remove points of data at random to check behaviour of neural network prediction and uncertainty.
def remove_data(X, Y, remove_lower, remove_upper):
    '''
    :param X: dataset for X
    :param Y: dataset for Y
    :param remove_lower: lower bound
    :param remove_upper: upper bound
    :return: new datasets
    '''
    X = X.flatten()
    Y = Y.flatten()

    X_remove = np.where(np.logical_and(X > remove_lower, X < remove_upper))

    for i in X_remove:
        X = np.delete(X, i)
        Y = np.delete(Y, i)

    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)

    return X, Y

