"""
prediction_likelihood calculates and saves the log-likelihood of the prediction
"""
import numpy as np 
import sys

sys.path.append('../')
import library_1d
import os

# taking theta range from library module
theta_min = library_1d.theta_min
theta_max = library_1d.theta_max

iterations = [1,2,3]
    
for n_iterations in iterations:
    sample_list = [10,20, 50, 100,200,300]

    for num in sample_list: 
        loc_1 = f'main_folder_location/single qubit/{num} samples/'    
        
        ensemble_sizes_list = [3,5,7,10,12,15]
        for model_num in ensemble_sizes_list:
            
            loc = loc_1 + f"repetition {n_iterations}/{model_num} models"
        
            path_true = os.path.join(loc, "true_f.txt")
            path_pred = os.path.join(loc, 'predicted_f.txt')
            path_std = os.path.join(loc, 'std.txt')
# =============================================================================
#             path_std = os.path.join(loc, 'std_capped.txt')
# =============================================================================
            path_theta = os.path.join(loc, 'data/theta_values.txt')
         #   path_theta = os.path.join(loc, "data_theta_values.txt")
            
            
            # these are shape (1000,) arrays
            true_f = np.loadtxt(path_true)
            pred_f = np.loadtxt(path_pred)
            std = np.loadtxt(path_std)
            
            num_theta = np.shape(true_f)[0] # this is number of points theta is split into = 1000
            
            def probability(true_f, pred_f, std, A):
                """
                Parameters
                ----------
                true_f : float
                pred_f : float
                std : float
                A : an adjustable constant to normalise the gaussian
            
                Returns
                -------
                the probability of obtaining true_f value given a gaussian centered around pred_f value
            
                """
                exponent = - 0.5 * (true_f - pred_f)**2/(std**2)
                probability = A * np.exp(exponent)
                
                return probability
                
            # empty list for appending probabilities    
            probabilities = []
            
            for i in range(num_theta):
                
                probability_i = probability(true_f[i], pred_f[i], std[i], A = 1)
                
                probabilities.append(probability_i)
                # appends this onto probabilities list which ends up being a 1000 entries list
                
            # define log likelihood 
            log_likelihood = np.log(np.prod(probabilities))
            print("The log-likelihood for the entire landscape is: ", log_likelihood)
        
            # save the values
            
            path_txt = os.path.join(loc, "error_summary.txt")
            
            with open(path_txt, "a") as text_file:
                print(f"Log-likelihood considering all theta: {log_likelihood}", file=text_file)
