"""
mean deviation and contour plot of deviation 

mean deviation should output a single value for delta z bar over entire plane

contour plot will be the delta z at every point
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import seaborn as sns
import shutil 
sys.path.append('../')
import library_2d
import two_dimensional_function as two_d 
import os

theta_1_min = library_2d.theta_1_min
theta_1_max = library_2d.theta_1_max
theta_2_min = library_2d.theta_2_min
theta_2_max = library_2d.theta_2_max

n_iterations = 1
while n_iterations <= 1:
    
    models_list = [1]
    for num_models in models_list:
        loc = f"C:/Users/Zhiyi/Desktop/year_3_project/0315data/repetition {n_iterations}/{num_models} models"
        
        path_true = os.path.join(loc, "true_f.txt")
        path_pred = os.path.join(loc, 'predicted_f.txt')
        path_std = os.path.join(loc, 'std.txt')
    
        true_f = np.loadtxt(path_true)
        predicted_f = np.loadtxt(path_pred)
        std = np.loadtxt(path_std)
        
        n_linspace = np.shape(predicted_f)[0]
        num_z = np.shape(predicted_f)[0] ** 2
        
        delta_z_array = []
        for j in range(n_linspace):
            row = []
            for i in range(n_linspace):
                deviation_ij = abs(predicted_f[i][j] - true_f[i][j])
                row.append(deviation_ij)
            delta_z_array.append(row)
            
        delta_z_array = np.array(delta_z_array)
        delta_z_flatten = delta_z_array.flatten()
        
# =============================================================================
#         getting array equivalent to delta_z_array but divide each delta z by std values
# =============================================================================
        delta_z_array_rel = []
        for j in range(n_linspace):
            row_2 = []
            for i in range(n_linspace):
                rel_deviation_ij = abs(predicted_f[i][j] - true_f[i][j])/std[i][j]
                row_2.append(rel_deviation_ij)
            delta_z_array_rel.append(row_2)
        
        delta_z_array_rel = np.array(delta_z_array_rel)
        
        sum_dev_over_std = np.sum(delta_z_flatten/std.flatten())
        
        # constructing the two types of errors
        error_dev_rel = sum_dev_over_std/num_z
        error_dev_abs = np.mean(delta_z_flatten)
        
        # constructing log-likelihood
        A_constant = 1
        exponent = - 0.5 * delta_z_array ** 2
        probabilities = A_constant * np.exp(exponent)
        log_likelihood = np.log(np.prod(probabilities))
        
        print ("Relative mean deviation error of landscape over the entire 2d range:", error_dev_rel)
        print ("Absolute mean deviation error of landscape over the entire 2d range:", error_dev_abs)
        print ("The log-likelihood of landscape over the entire 2d range: ", log_likelihood)
        
        path_txt = os.path.join(loc, "error_summary.txt")
        shutil.rmtree(path_txt, ignore_errors = True)
        
        with open(path_txt, "w") as text_file:
            print(f"Mean deviation error relative to one std: {error_dev_rel}", file=text_file)
            print(f"Mean deviation error absolute: {error_dev_abs}", file=text_file)
            print(f"log-likelihood: {log_likelihood}", file=text_file)
            
# =============================================================================        
#        plotting contour plot for the deviation for each (theta_1, theta_2) point
#        - tricontour plotting
# =============================================================================
        loc_theta = f"C:/Users/Zhiyi/Desktop/year_3_project/0315data/repetition {n_iterations}/data/"
        
        theta_1 = np.loadtxt(loc_theta+"theta_1_values.txt")
        theta_2 = np.loadtxt(loc_theta+"theta_2_values.txt")
        
        sns.set_style("whitegrid")
        fig, ax = plt.subplots()        
        
        ax.contour(theta_1, theta_2, delta_z_array, levels=10, linewidths=0.5, colors='b')
        contour = ax.contourf(theta_1, theta_2, delta_z_array, levels=15, cmap="RdBu_r")
        
        fig.colorbar(contour, ax=ax)
        ax.set(xlim=(theta_1_min, theta_1_max), ylim=(theta_2_min, theta_2_max))
        ax.set_xlabel(r'$\theta_{1}$', fontsize=22)
        ax.set_ylabel(r'$\theta_{2}$', fontsize=22)
        ax.set_title("Absolute Prediction Deviation from True Landscape", fontsize=22)
        plt.savefig(loc+"/abs_dev_contour_plot.png", dpi=200) 
        
        fig2, ax2 = plt.subplots()
        ax2.contour(theta_1, theta_2, delta_z_array, levels=10, linewidths=0.5, colors='b')
        contour2 = ax2.contourf(theta_1, theta_2, delta_z_array_rel, levels=15, cmap="RdBu_r")
        
        fig2.colorbar(contour2, ax=ax2)
        ax2.set(xlim=(theta_1_min, theta_1_max), ylim=(theta_2_min, theta_2_max))
        ax2.set_xlabel(r'$\theta_{1}$', fontsize=22)
        ax2.set_ylabel(r'$\theta_{2}$', fontsize=22)
        ax2.set_title("Relative Prediction Deviation from True Landscape", fontsize=22)
        plt.savefig(loc+"/rel_dev_contour_plot.png", dpi=200)
        
# =============================================================================
#         3d contour plot of the relative deviation
#         tbh just a 3d representation of the above
# =============================================================================
        fig3 = plt.figure()
        ax3 = plt.axes(projection='3d')
        ax3.contour3D(theta_1, theta_2, delta_z_array_rel, 200, cmap="RdBu_r")
        
        ax3.set_xlabel(r'$\theta_{1}$',fontsize=20)
        ax3.set_ylabel(r'$\theta_{2}$',fontsize=20)
        ax3.set_zlabel(r"Deviation",fontsize=20)
        ax3.xaxis.labelpad=10
        ax3.yaxis.labelpad=10
        ax3.zaxis.labelpad=10
        plt.title("Relative Prediction Deviation from True Landscape (w.r.t. 1 std)", fontsize=20)
        plt.savefig(loc+"/rel_dev_3d_contour.png", dpi=200)
                


    n_iterations += 1