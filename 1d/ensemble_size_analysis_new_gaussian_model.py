"""
the gaussian model in a while loop - also puts results for different ensemble sizes inside
repetition folders 

for loop within while loop
"""
import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
sys.path.append('../')
import library
from contextlib import redirect_stdout



n_iterations = 1
while n_iterations <= 3:
    
    iteration_folder = f'main_folder_location/repetition {n_iterations}/'
    
    ensemble_sizes_list = [1,2,3,5,7,10,12,15]

    for model_num in ensemble_sizes_list:
        print("Iteration ", n_iterations, "- Ensemble Size: ", model_num)
            # log errors to a .txt file

        
        #disable eager execution (a mode within tensorflow)
        tf.compat.v1.disable_eager_execution()
        
        #load command line inputs
        args = sys.argv
            
        try:
            toymodel = args[1]
            n_samples = int(args[2])
            n_models = int(args[3])
            model_name = args[4]
            noise = args[5:]
        except IndexError:
            toymodel = "single_qubit"
            n_samples = 300
            n_models = model_num
            model_name = f"Single Qubit ({model_num} models)"
            noise = "g", 0, 0.05
            
    
        
        # noise can be none ('n'), Gaussian ('g') or binomial ('b') with appropriate parameters
        #noise = sys.argv[5:]
        noise_str = library.noise(noise)
        
        #take in range constraints from the library
        theta_min = library.theta_min
        theta_max = library.theta_max
        
        # specifying iteration folder as the folder to go look
        data_path, results_path, model_path, loss_curves_path = library.model_pathways_make_directories_regular(iteration_folder, model_num, toymodel, model_name, n_samples, noise)
        
        # load the data
        X = np.loadtxt(os.path.join(data_path, f'x-{n_samples}{noise_str}.csv'), dtype=np.float64)
        Y = np.loadtxt(os.path.join(data_path, f'y-{n_samples}{noise_str}.csv'), dtype=np.float64)
        
        #reshape data such that they're column vectors 
        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)
        
        # define the number of epochs and batch size for training
        epochs = 2000
        batch_size = 100
        
        # the number of hidden units in a one layer model
        n_hidden_units = 50
        
        # the learning rate for the optimizer
        lr = 1e-4 
        

        #standardise data so it all members lie between -1 and 1 allowing for better performance
        
        xnormalise = library.Normaliser()
        X = xnormalise.standardise(X)
        
        ynormalise = library.Normaliser()
        Y = ynormalise.standardise(Y)
        
        
        
        
        def one_layer_model_nll(n):
            """
            :param n: the number of hidden units of the one layer model
            :return: the one layer model with a nll loss function
            """
            inputs = tf.keras.Input(shape=(1,), name='input')
        
            hidden_layer = tf.keras.layers.Dense(n, activation='relu', name='hidden_layer')(inputs)
        
            output_mu, output_var = library.OutputLayer(1, name='output')(hidden_layer)
        
            model = tf.keras.Model(inputs=inputs, outputs=output_mu, name=model_name)
        
            adam = tf.keras.optimizers.Adam(learning_rate=lr)
        
            model.compile(loss=library.custom_loss(output_var), optimizer=adam)
        
            return model
        
        
        def small_model():
            '''
            :return: a model with 3 hidden layers and two outputs for mean and variance
            '''
            inputs = tf.keras.Input(shape=(1,), name='input')
            hidden_layer = tf.keras.layers.Dense(50, activation='relu', name='hidden_layer0')(inputs)
            hidden_layer = tf.keras.layers.Dense(100, activation='relu', name='hidden_layer1')(hidden_layer)
            hidden_layer = tf.keras.layers.Dense(50, activation='relu', name='hidden_layer2')(hidden_layer)
            output_mu, output_var = library.OutputLayer(name='output')(hidden_layer)
        
            model = tf.keras.Model(inputs=inputs, outputs=output_mu, name=model_name)
            # changed learning rate value to lr instead
            adam = tf.keras.optimizers.Adam(learning_rate=lr)
            model.compile(loss=library.custom_loss(output_var), optimizer=adam)
        
            # tf.keras.utils.plot_model(model, os.path.join(model_path, 'model.png'), show_shapes=True)
        
            return model
        
        
        models = []
        
        for i in range(n_models):
        
            model = small_model()
        
            if i == 1:
                model_summary_path = os.path.join(model_path, 'model_summary.txt')
                with open(model_summary_path, 'w') as file:
                    with redirect_stdout(file):
                        model.summary()
            
            print("Iteration ", n_iterations, "- Ensemble Size: ", model_num)
            print("Model Number", i+1, "of", n_models)
            combined = np.concatenate((X.reshape(-1,1), Y.reshape(-1,1)), axis=1)
        #    combined = np.concatenate((X, Y), axis=1)
            # combines them as column vector of [x,y] pairs and shuffles the pairs
            np.random.shuffle(combined)
        
            X_train = combined[:, 0]
            Y_train = combined[:, 1]
        
            history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,
                                shuffle=True, validation_split=0.2)
            # model.fit trains the data and holds the results 
            models.append(model)
        
            mpl.rcParams['figure.figsize'] = [12.8, 9.6]
            mpl.rcParams.update({'font.size': 22})
            sns.set_style(style='darkgrid')
        
            plt.figure()
            pd.DataFrame(history.history).plot()
            # above plots the history variable (dictionary) that is converted to a pandas dataframe
            plt.title(f'Model Name: {model_name}-{i + 1}')
            plt.xlabel(f'Epoch (batch size={batch_size})')
            plt.savefig(os.path.join(loss_curves_path, f'{model_name}-{i + 1}'))
            plt.close()
            
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title(f'Model {i+1}: Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['training', 'validation'], loc='upper left')
            plt.show()
            
# =============================================================================
#             if validation loss is greater than training loss then model is overfitting
#             if validation loss is less than training loss then model is underfitting
# =============================================================================
        
            model.save(os.path.join(model_path, f'{model_name}-{i + 1}.h5'))
            del model
        
        #redefine the range for cubic mostly for plotting purposes
        if toymodel=='cubic':
            theta_min = theta_min - 2
            theta_max = theta_max + 2
        
        num_theta = 1000
        
        theta_not_norm = np.linspace(theta_min, theta_max, num=num_theta).reshape(-1, 1)
        # this is in the form of a column vector
        
        # standardise the x data
        theta = xnormalise.standardise2(theta_not_norm)
        
        predicted_f_ensemble = np.zeros((n_models, num_theta))
        variance_ensemble = np.zeros((n_models, num_theta))
        
        # Generate an ensemble of (number_of_models) models
        for i, model in enumerate(models):
            model = models[i]
            
            # tensor function that is outputted from the output layer
            get_predicted = K.function(inputs=[model.input], outputs=model.get_layer('output').output)
            
            # the tensor function with standardised theta as arguments 
            predicted = get_predicted(theta)
            
            #predicted is 2 column vectors 
            
            predicted_f = predicted[0].flatten()
            # flattens the column vector into a row/list
            predicted_f_ensemble[i] += predicted_f
            
            variance_f = predicted[1].flatten()
            variance_ensemble[i] += variance_f
        
        # below is the mean of predicted f
        mu_star = np.mean(predicted_f_ensemble, axis=0)
        # below is variance on predicted_f_ensemble?
        
        mu_star_discrete_variance = (np.sum(predicted_f_ensemble ** 2, axis=0) / n_models) - (mu_star ** 2)
        var_star = (np.sum(variance_ensemble, axis=0) / n_models) + mu_star_discrete_variance
        # var star has two contributions 
            # 1. the variance average contribution from individual ANNs 
            # 2. the mu star variance contribution 
            #   (this is mu average contrbution from individual ANNs squared - mu_star**2 )
        
        true_f = library.single_qubit(theta_not_norm.flatten())
        
        theta = theta.flatten()
        predicted_f = ynormalise.destandardise(mu_star)
        std = ynormalise.destandardise_var(var_star) ** .5
        
        X = xnormalise.destandardise(X)
        Y = ynormalise.destandardise(Y)
        
        n_std = 1.96
        
        plt.figure()
        
        plt.plot(theta_not_norm, true_f, linestyle='--', color='black', label='true')
        plt.plot(theta_not_norm, predicted_f, linestyle='--', color='red', label='predicted')
        plt.plot(X, Y, '*', color='black', label='samples')
        plt.fill_between(theta_not_norm.flatten(), predicted_f - (n_std * std), predicted_f + (n_std * std), color='gray',
                         alpha=0.5, label=f'{n_std} standard deviations')
        
        plt.title(f'{model_name}: {n_samples} examples')
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$f(\theta)$')
        plt.legend()
        
        plt.savefig(os.path.join(model_path, f'curves-{n_samples}'))
        
        
        """
        saving required inputs for deviation_error.py and prediction_likelihood.py
        - true f (array of floats (1000,))
        - predicted f (array of floats (1000,))
        - std (array of floats (1000,))
        
        a.k.a. all column vectors 
        
        """
        results_ensemble_size_folder = os.path.join(iteration_folder, f"{model_num} models")
    
        path_true = os.path.join(results_ensemble_size_folder, "true_f.txt")
        path_pred = os.path.join(results_ensemble_size_folder, 'predicted_f.txt')
        path_std = os.path.join(results_ensemble_size_folder, 'std.txt')
    
        np.savetxt(path_true, true_f)
        np.savetxt(path_pred, predicted_f)
        np.savetxt(path_std, std)

        # save arguments 
        
        labels = np.array(["Number of Samples", "Number of Models", "Toy Model", 
                           "Noise information", "Epoch Size", "Batch Size", 
                           "Learning Rate"]).reshape(-1,1)
        values = np.array([n_samples, model_num, toymodel, noise, epochs, batch_size, 
                           lr]).reshape(-1,1)
        
        arguments = np.concatenate((labels, values), axis = 1)
        
        path_arguments = os.path.join(results_ensemble_size_folder, 'arguments.txt')
        
        np.savetxt(path_arguments, arguments, fmt='%5s',delimiter=': ')  
        
    n_iterations += 1