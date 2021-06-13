# -*- coding: utf-8 -*-
"""
prediction_likelihood with while loop
"""
import numpy as np 
import sys

sys.path.append('../')
import library
import os

# taking theta range from library module
theta_min = library.theta_min
theta_max = library.theta_max

iterations = [1,2,3]
    
for n_iterations in iterations:
    sample_list = [10,20, 50, 100,200,300]

    for num in sample_list: 
        loc_1 = f'C:/Users/xious/Documents/Year 3/BSc Project/Quantifying accuracy/single qubit/{num} samples/'    
        
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
        
            # save the values"
            
            path_txt = os.path.join(loc, "error_summary.txt")
            
            with open(path_txt, "a") as text_file:
                print(f"Log-likelihood considering all theta: {log_likelihood}", file=text_file)
           #     print(f"Log-likelihood considering only data theta: {log_likelihood_samp}", file=text_file)    

    
#%%
""""        
    
    """
  #  the following code is for the purpose of only getting probability at the theta values
   # corresponding to the sample data points 
        
"""   
    sample_thetas = np.loadtxt(path_theta) # this is shape (n_samples,) 
      
    from bisect import bisect_left
    
    def take_closest(List, Number):
        """
    #    Assumes List is sorted. Returns the (list index of the) closest number to the Number.
"""
        index = bisect_left(List, Number)
        
      #  num_in_list = List[index]
        
        return index
        
    # need new true_f and pred_f at the points of theta in sample_thetas
    
    new_true_f = []
    new_pred_f = []
    new_std = []
    
    
    num_theta_data = np.shape(sample_thetas)[0]
    
    x_theta = np.linspace(0, 4, 1000)
    
    """
  #  trying to return list of (n_samples) indices that correspond to the sample_thetas within the
 #   1000 x_values 
"""
    data_indices = []
    
    for i in range(num_theta_data):
        
        data_indices_i = take_closest(x_theta, sample_thetas[i])
        
        data_indices.append(data_indices_i)
        # data_indices now has n_samples elements 
    
    
    # now need to rip out the values of the true and pred f corresponding to indices in data_indices
    for i in range(num_theta_data):
        
        # taking the y values corresponding to the indices given in the list of data_indices
        new_true_f_i = true_f[data_indices[i]]
        new_pred_f_i = pred_f[data_indices[i]]
        new_std_i = std[data_indices[i]]
        
        new_true_f.append(new_true_f_i)
        new_pred_f.append(new_pred_f_i)
        new_std.append(new_std_i)
            
    # now all above stuff but for only considering sample data points 
    
    
    def probability_samp(new_true_f, new_pred_f, new_std, A):
        """
       # same as probability but with only values corresponding to the sample thetas 
"""
        exponent = - 0.5 * (new_true_f - new_pred_f)**2/(new_std**2)
        probability = A * np.exp(exponent)
        
        return probability
        
    # empty list for appending probabilities    
    probabilities_samp = []
    
    for i in range(num_theta_data):
        
        probability_samp_i = probability_samp(new_true_f[i], new_pred_f[i], std[i], A = 1)
        
        probabilities_samp.append(probability_samp_i)
        # appends this onto probabilities list which ends up being a 1000 entries list
        
    # define log likelihood 
    log_likelihood_samp = np.log(np.prod(probabilities_samp))
    print("The log-likelihood considering sample points only is: ", log_likelihood_samp)
    """