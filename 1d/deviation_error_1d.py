"""
This script is for finding the theta integral error/deviation error:
    
    theta_integral_error = (integral over theta range of 
                            mod(deviation of predicted from true)/std)/(theta range) (or without std)
    
    where std is the standard deviation used to plot the confidence interval
    
    We approx. this integral to a sum since theta and the prediction is not
    defined as a continuous function but rather 1000 discrete points, it's
    a measure of the average deviation between the predicted landscape from the true one.
"""
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import shutil 
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
        
        ensemble_sizes_list = [1, 2]
        for model_num in ensemble_sizes_list:
            
            loc = loc_1 + f"repetition {n_iterations}/{model_num} models"
   
            path_true = os.path.join(loc, "true_f.txt")
            path_pred = os.path.join(loc, 'predicted_f.txt')
            path_std = os.path.join(loc, 'std.txt')
# =============================================================================
#             path_std = os.path.join(loc, 'std_capped.txt')
# =============================================================================
            
            # these are shape (1000,) arrays
            true_f = np.loadtxt(path_true)
            predicted_f = np.loadtxt(path_pred)
            std = np.loadtxt(path_std)
            
            num_theta = np.shape(true_f)[0]
            
            deviation = abs(predicted_f - true_f)
            sum_dev = np.sum(deviation)
            sum_dev_over_std = np.sum(deviation/std)
            
            # constructing the two types of errors
            error_dev_rel = sum_dev_over_std/num_theta
            error_dev_abs = sum_dev/num_theta
            
            print ("Relative mean deviation error of landscape over the entire theta range:", error_dev_rel)
            print ("Absolute mean deviation error of landscape over the entire theta range:", error_dev_abs)
            
            # can try and plot what the deviation is for every theta
            
            x_theta = np.linspace(theta_min, theta_max, num = 1000)
            
            sns.set_style("darkgrid")
            fig, axes = plt.subplots(nrows=2)
            
            axes[0].plot(x_theta, deviation, "green", linewidth = 2.5)
            axes[0].set_xlabel(r'$\theta$')
            axes[0].set_ylabel("Absolute Deviation")
            
            axes[1].plot(x_theta, deviation/std, "blue", linewidth = 2.5)
            axes[1].set_xlabel(r'$\theta$')
            axes[1].set_ylabel("Relative Deviation (w.r.t 1 std)")
            
            axes[0].set_title("Deviation Error of Landscape Prediction")
            plt.tight_layout()
            
            path_fig = os.path.join(loc, "Plot - Deviation Error of Landscape Prediction")
            plt.savefig(path_fig)
            
            # save the deviation error
        
        
            path_txt = os.path.join(loc, "error_summary.txt")
            shutil.rmtree(path_txt, ignore_errors = True)
            
            with open(path_txt, "w") as text_file:
                print(f"Mean deviation error relative to one std: {error_dev_rel}", file=text_file)
                print(f"Mean deviation error absolute: {error_dev_abs}", file=text_file)
