"""
gaussian model for 2d
"""
import shutil 
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
import library_2d
from contextlib import redirect_stdout
import matplotlib.ticker as tck
from matplotlib.patches import Patch

# repeats using differents sets of data(with different random seeds for noise)

n_iterations = 1
while n_iterations <= 3:
    
    iteration_folder = f'C:/Users/Zhiyi/Desktop/year_3_project/0315data/repetition {n_iterations}/'
    
    ensemble_sizes_list = [1,2,3,5,7,10,12,15]

    for model_num in ensemble_sizes_list:
        print("Iteration ", n_iterations, "- Ensemble Size: ", model_num)
            # log errors to a .txt file
        #library.error_handling(sys.argv[0][:-3])
        
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
            toymodel = "qubit"
            n_samples = 300
            n_models = model_num
            model_name = "test"
            noise = "g", 0, 0.05
            
    
        
        # noise can be none ('n'), Gaussian ('g') or binomial ('b') with appropriate parameters
        noise_str = library_2d.noise(noise)
        
        #take in range constraints from the library
        theta_1_min = library_2d.theta_1_min
        theta_1_max = library_2d.theta_1_max
        theta_2_min = library_2d.theta_2_min
        theta_2_max = library_2d.theta_2_max
        
        # specifying iteration folder as the folder to go look
        data_path, results_path, model_path, loss_curves_path = library_2d.model_pathways_make_directories_regular(iteration_folder, model_num, toymodel, model_name, n_samples, noise)
        
        # load the data
        X = np.loadtxt(os.path.join(data_path, f'x-{n_samples}{noise_str}.csv'), dtype=np.float64, delimiter=',')
        Y = np.loadtxt(os.path.join(data_path, f'y-{n_samples}{noise_str}.csv'), dtype=np.float64, delimiter=',')
        Z = np.loadtxt(os.path.join(data_path, f'z-{n_samples}{noise_str}.csv'), dtype=np.float64, delimiter=',')
        
        
        #reshape data such that they're column vectors 
        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)
        X_orig = X.reshape(-1, 1)
        Y_orig = Y.reshape(-1, 1)
        Z_orig = Z.reshape(-1, 1)
        
        # define the number of epochs and batch size for training
        epochs = 500
        batch_size = 100
        
        # the number of hidden units in a one layer model
        n_hidden_units = 50
        
        # the learning rate for the optimizer
        lr = 1e-4 # reduced from 1e-3
        
        # the seed for numpy.random to use when generating the data
        seed = n_iterations
        
        # delete the below indexed data point
        #X = np.delete(X, np.argmax(Y))
        #Y = np.delete(Y, np.argmax(Y))
         
        """
        # remove within x range
         X, Y = library.remove_data(X,Y,1.5,2.4)
        """
        
        #standardise data so it all members lie between -1 and 1 allowing for better performance
        
        xnormalise = library_2d.Normaliser()
        X = xnormalise.standardise(X)
        
        ynormalise = library_2d.Normaliser()
        Y = xnormalise.standardise(Y)
        
        znormalise = library_2d.Normaliser()
        Z = ynormalise.standardise(Z)
        
        
        def one_layer_model_nll(n):
            """
            :param n: the number of hidden units of the one layer model
            :return: the one layer model with a nll loss function
            """
            inputs = tf.keras.Input(shape=(2,), name='input')
        
            hidden_layer = tf.keras.layers.Dense(n, activation='relu', name='hidden_layer')(inputs)
        
            output_mu, output_var = library_2d.OutputLayer(1, name='output')(hidden_layer)
        
            model = tf.keras.Model(inputs=inputs, outputs=output_mu, name=model_name)
        
            adam = tf.keras.optimizers.Adam(learning_rate=lr)
        
            model.compile(loss=library_2d.custom_loss(output_var), optimizer=adam)
        
            # tf.keras.utils.plot_model(model, os.path.join(model_path, 'model.png'), show_shapes=True)
        
            return model
        
        
        def small_model():
            '''
            :return: a model with 3 hidden layers and two outputs for mean and variance
            '''
            inputs = tf.keras.Input(shape=(2,), name='input')
            hidden_layer = tf.keras.layers.Dense(50, activation='relu', name='hidden_layer0')(inputs)
            hidden_layer = tf.keras.layers.Dense(100, activation='relu', name='hidden_layer1')(hidden_layer)
            hidden_layer = tf.keras.layers.Dense(100, activation='relu', name='hidden_layer2')(hidden_layer)
            hidden_layer = tf.keras.layers.Dense(100, activation='relu', name='hidden_layer3')(hidden_layer)
            hidden_layer = tf.keras.layers.Dense(100, activation='relu', name='hidden_layer4')(hidden_layer)
            hidden_layer = tf.keras.layers.Dense(100, activation='relu', name='hidden_layer5')(hidden_layer)
            hidden_layer = tf.keras.layers.Dense(100, activation='relu', name='hidden_layer6')(hidden_layer)
            hidden_layer = tf.keras.layers.Dense(50, activation='relu', name='hidden_layer7')(hidden_layer)
           
            output_mu, output_var = library_2d.OutputLayer(name='output')(hidden_layer)
        
            model = tf.keras.Model(inputs=inputs, outputs=output_mu, name=model_name)
            # changed learning rate value to lr instead
            adam = tf.keras.optimizers.Adam(learning_rate=lr)
            model.compile(loss=library_2d.custom_loss(output_var), optimizer=adam)
        
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


            combined_xy = []
            
            X = np.array(X).reshape(-1,1).tolist()
            Y = np.array(Y).reshape(-1,1).tolist()
            Z = np.array(Z).reshape(-1,1).tolist()


            combined_xyz = np.concatenate((X,Y,Z), axis = 1)
            
            np.random.shuffle(combined_xyz) # shape looks like (n_samples**2, 3)
            # where the first 2 columns are the xy pairs to give the z in 3rd column
            
            
            XY_train = combined_xyz[:,:2] # makes input (2,)
            Z_train = combined_xyz[:,2]
    
        
            history = model.fit(XY_train, Z_train, epochs=epochs, batch_size=batch_size,
                                shuffle=True, validation_split=0.2)
            # model.fit trains the data and holds the results 
            models.append(model)
        
            mpl.rcParams['figure.figsize'] = [12.8, 9.6]
            mpl.rcParams.update({'font.size': 22})
            sns.set_style(style='darkgrid')
        
            plt.figure()
            pd.DataFrame(history.history).plot()
            # above plots the history variable (dictionary) that is converted to a pandas dataframe
            # hist_df = pd.DataFrame(history.history)
            plt.title(f'Model Name: {model_name}-{i + 1}')
            plt.xlabel(f'Epoch (batch size={batch_size})')
            plt.savefig(os.path.join(loss_curves_path, f'{model_name}-{i + 1}'))
            plt.close()
        
            model.save(os.path.join(model_path, f'{model_name}-{i + 1}.h5'))
            
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title(f'Model {i+1}: Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['training', 'validation'], loc='upper left')
            plt.show()
            del model
        
        #redefine the range for cubic mostly for plotting purposes
# =============================================================================
#         if toymodel=='cubic':
#             theta_min = theta_min - 2
#             theta_max = theta_max + 2
# =============================================================================
        
        # this defines the linspace range
        num_theta = 100
        
        theta_1_not_norm = np.linspace(theta_1_min, theta_1_max, num=num_theta).reshape(-1, 1)
        theta_2_not_norm = np.linspace(theta_2_min, theta_2_max, num=num_theta).reshape(-1, 1)
        # this is in the form of a column vector
        
        # standardise the x data
        theta_1 = xnormalise.standardise2(theta_1_not_norm)
        theta_2 = xnormalise.standardise2(theta_2_not_norm)
                    
        
        theta_tuple_matrix = []
        for j in theta_2:
            for i in theta_1:
                ij = (i,j)
                theta_tuple_matrix.append(ij)
                
        theta_tuple_matrix = np.array(theta_tuple_matrix)
        
        
        predicted_f_ensemble = np.zeros((n_models, num_theta**2))
        variance_ensemble = np.zeros((n_models, num_theta**2))
        
        # Generate an ensemble of (number_of_models) models
        for i, model in enumerate(models):
            model = models[i]
            
            # tensor function that is outputted from the output layer
            get_predicted = K.function(inputs=[model.input], outputs=model.get_layer('output').output)
            
            # the tensor function with matrix of thetas as arguments 
            predicted = get_predicted(theta_tuple_matrix.reshape(-1,2))
            
            #predicted is 2 column vectors 
            
            predicted_f = predicted[0].flatten()
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
        

        
        true_f = [] # ends up a (n_samples, n_samples) matrix
        for i in range(len(theta_2_not_norm)):
            row=[]
            for j in range(len(theta_1_not_norm)):
                f = library_2d.function(theta_1_not_norm[j], theta_2_not_norm[i])
                row.append(f)
            true_f.append(row)
            


        #reshape and destadradise the data for plotting
        theta_1 = theta_1.flatten()
        theta_2 = theta_2.flatten()
    
        predicted_f = ynormalise.destandardise(mu_star).reshape(num_theta,num_theta).tolist()
        std = ynormalise.destandardise_var(var_star) ** .5
        std = std.reshape(num_theta,num_theta)
        
        X = np.array(X).flatten()
        Y = np.array(Y).flatten()
        Z = np.array(Z).flatten()
        
        X = xnormalise.destandardise(X)
        Y = xnormalise.destandardise(Y)
        Z = ynormalise.destandardise(Z)
        
        n_std = 1.96
        

        sns.set_style("whitegrid")
# =============================================================================
#         the 3d contour plot
# =============================================================================
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        surf_true = ax.contour3D(theta_1_not_norm.flatten(), theta_2_not_norm.flatten(),true_f, 100, label='true landscape', cmap='binary')
        surf_predicted = ax.contour3D(theta_1_not_norm.flatten(), theta_2_not_norm.flatten(),predicted_f, 100, label='predicted landscape', cmap='inferno')
        ax.scatter(X, Y, Z, '*', color='black',zorder=1)
        ax.set_xlabel(r'$\theta_{1}$',fontsize=20)
        ax.set_ylabel(r'$\theta_{2}$',fontsize=20)
        

        plt.xticks([0, 1/4*np.pi, 1/2*np.pi, 3/4*np.pi, np.pi], [0, r'$\pi$/4', r'$\pi$/2', r'3$\pi$/4', r'$\pi$'])  # Set text labels.
        plt.yticks([0, 1/4*np.pi, 1/2*np.pi, 3/4*np.pi, np.pi], [0, r'$\pi$/4', r'$\pi$/2', r'3$\pi$/4', r'$\pi$']) 
        
        custom_lines = [Patch(facecolor='dimgrey', edgecolor='black',linewidth=3,
                         label='true'),
                        Patch(facecolor='darkmagenta', edgecolor='orange',linewidth=3,
                         label='predicted')]
        
        ax.legend(custom_lines, ['true','predicted'],loc=1, prop={'size': 18})
        
        ax.set_zlabel(r"$|F(\theta_{1}, \theta_{2})|^{2}$",fontsize=20)
        ax.xaxis.labelpad=20
        ax.yaxis.labelpad=20
        ax.zaxis.labelpad=20
        plt.title(r'$\sigma_{x}$$\sigma_{y}$ combination: ' + f'{n_samples} examples', fontsize=20)
        plt.savefig(os.path.join(model_path, f'contour-{n_samples}'), dpi=200)
        
        plt.figure()
    
# =============================================================================
#         the comparison w standard deviation plot
# =============================================================================
        plt.plot(theta_2_not_norm.flatten(), np.array(predicted_f)[:,49], color='red', label='predicted')
        plt.plot(theta_2_not_norm.flatten(), np.array(true_f)[:,49], color='black', label='true')
        plt.fill_between(theta_2_not_norm.flatten(), 
                         np.array(predicted_f)[:,49] - (n_std * std)[:,49], 
                         np.array(predicted_f)[:,49] + (n_std * std)[:,49], color='gray',
                          alpha=0.5, label=f'{n_std} standard deviations')
        
        plt.plot(Y_orig, Z_orig[:,0], '*', color='black', label=r'samples ($\theta_{1}$ = 0 plane projection)')
        
        plt.title(r'$\sigma_{x}$$\sigma_{y}$ combination, $\theta_{1}$ = 0 plane: ' + f'{n_samples} examples',fontsize=20)
        plt.xlabel(r'$\theta_{2}$', fontsize=20)
        plt.ylabel(r'$f(\theta_{2})$', fontsize=20)
        plt.savefig(os.path.join(model_path, f'curves-{n_samples}'), dpi=200)
        plt.legend(fontsize=15)
        

        
# =============================================================================
#         only reason why this is so good is because it has the other points in the plane
#         as reference, we just don't plot them, so averaging out it's got to 
#         give a good prediction for a landscape that only varies in one theta, 
#         for landscapes that vary in both thetas, might not be so good
# =============================================================================
        
        
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

        labels = np.array(["Number of Samples", "Number of Models", "Toy Model", 
                           "Noise information", "Epoch Size", "Batch Size", 
                           "Learning Rate"]).reshape(-1,1)
        values = np.array([n_samples, model_num, toymodel, noise, epochs, batch_size, 
                           lr]).reshape(-1,1)
        
        arguments = np.concatenate((labels, values), axis = 1)
        
        path_arguments = os.path.join(results_ensemble_size_folder, 'arguments.txt')
        
        np.savetxt(path_arguments, arguments, fmt='%5s',delimiter=': ') 
        
    n_iterations += 1

