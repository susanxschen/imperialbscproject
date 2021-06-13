"""
effectively the same as plot_error_against_ensemble_size.py but plotting 4 
different landscape results on one graph for the three produced plots 
"""
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sys.path.append('../')
import os
import re


models_names = ["model_1", "model_2", "model_3",
                "model_4"]

ensemble_sizes = [3, 5, 7, 10, 12, 15, 18, 20]

n = 10 # iteration number

median_mean_dev_list_1 = []
median_mean_dev_list_2 = []
median_mean_dev_list_3 = []
median_mean_dev_list_4 = []

median_abs_dev_list_1 = []
median_abs_dev_list_2 = []
median_abs_dev_list_3 = []
median_abs_dev_list_4 = []

median_log_like_list_1 =[]
median_log_like_list_2 =[]
median_log_like_list_3 =[]
median_log_like_list_4 =[]

for name in models_names:
    
    loc = f'main_folder_location/{name}'
    
    
    # below i labels ensemble size, j labels iteration
    for i in ensemble_sizes:
        
        # lists defined 
        rel_deviation_list_i = []
        abs_deviation_list_i = []
        log_like_list_i = []
        for j in range(1, 11):
            if name != "Leo noiseless week 4":
                path_models_folder_ij = os.path.join(loc, f"repetition {j}/{i} models/")
            else:
                path_models_folder_ij = os.path.join(loc, f"{i} models/{j} iterations")
                
            
            # list of the relevant measures for the 10 iteration values for one ensemble size i
            
            path_error = os.path.join(path_models_folder_ij, "error_summary.txt")
            
            with open(path_error) as file:
                lines= []
                for line in file:
                    lines.append(line)
                    
                error_string = lines[0]
                abs_error_string = lines[1]
                log_like_string = lines[2]
                
                
            p = re.compile(r'\d+\.\d+')  # pattern to capture float values
            rel_mean_deviation_ij = [float(j) for j in p.findall(error_string)]
            abs_mean_deviation_ij = [float(j) for j in p.findall(abs_error_string)]
            
            #need to add minus here since it doesn't pick out the negative
            log_like_ij = [- float(j) for j in p.findall(log_like_string)]
            
            rel_deviation_list_i.append(rel_mean_deviation_ij)
            abs_deviation_list_i.append(abs_mean_deviation_ij)
            log_like_list_i.append(log_like_ij)
            
      
        median_rel_deviation_i = np.median(rel_deviation_list_i) 
        median_abs_deviation_i = np.median(abs_deviation_list_i)
        median_log_i = np.median(log_like_list_i)
            
        # appending mean, abs dev and log onto the main lists
        
        if name == "model_1":
            median_mean_dev_list_1.append(median_rel_deviation_i)
            median_abs_dev_list_1.append(median_abs_deviation_i)
            median_log_like_list_1.append(median_log_i)
            
        elif name == "model 3":
            median_mean_dev_list_3.append(median_rel_deviation_i)
            median_abs_dev_list_3.append(median_abs_deviation_i)
            median_log_like_list_3.append(median_log_i)
            
        elif name == "model 4":
            median_mean_dev_list_4.append(median_rel_deviation_i)
            median_abs_dev_list_4.append(median_abs_deviation_i)
            median_log_like_list_4.append(median_log_i)
            
        else:
            median_mean_dev_list_2.append(median_rel_deviation_i)
            median_abs_dev_list_2.append(median_abs_deviation_i)
            median_log_like_list_2.append(median_log_i)

    
loc_save = 'main_folder_location'

sns.set_style("darkgrid")

plt.figure()
plt.errorbar(x = ensemble_sizes, y = median_mean_dev_list_4,
             fmt="o-", color="red", ms = 5, label = "model 4")
plt.errorbar(x = ensemble_sizes, y = median_mean_dev_list_1,
             fmt="o-", color="blue", ms = 5, label = "model 1")
plt.errorbar(x = ensemble_sizes, y = median_mean_dev_list_2,
             fmt="o-", color="green", ms = 5, label = "model 2")
plt.errorbar(x = ensemble_sizes, y = median_mean_dev_list_3,
             fmt="o-", color="purple", ms = 5, label = "model 3")

plt.xlabel("Enesemble Size")
plt.ylabel("Median Mean Deviation (w.r.t. 1 std)")
plt.legend()
plt.title("Median Mean Deviation (w.r.t. 1 std) against Ensemble Size")
plt.savefig(os.path.join(loc_save, "Plot_rel_mean_dev_against_ensemble_size.png"), dpi=200)

plt.figure()
plt.errorbar(x = ensemble_sizes, y = median_abs_dev_list_4,
             fmt="o-", color="red", ms = 5, label = "model 4")
plt.errorbar(x = ensemble_sizes, y = median_abs_dev_list_1,
             fmt="o-", color="blue", ms = 5, label = "model 1")
plt.errorbar(x = ensemble_sizes, y = median_abs_dev_list_2,
             fmt="o-", color="green", ms = 5, label = "model 2")
plt.errorbar(x = ensemble_sizes, y = median_abs_dev_list_3,
             fmt="o-", color="purple", ms = 5, label = "model 3")

plt.xlabel("Enesemble Size")
plt.ylabel("Median Mean Deviation (Absolute)")
plt.legend()
plt.title("Median Mean Deviation (Absolute) against Ensemble Size")
plt.savefig(os.path.join(loc_save, "Plot_abs_mean_dev_against_ensemble_size.png"), dpi=200)

plt.figure()

plt.errorbar(x = ensemble_sizes, y = median_log_like_list_4,
             fmt="o-", color="red", ms = 5, label = "model 4")
plt.errorbar(x = ensemble_sizes, y = median_log_like_list_1,
             fmt="o-", color="blue", ms = 5, label = "model 1")
plt.errorbar(x = ensemble_sizes, y = median_log_like_list_2,
             fmt="o-", color="green", ms = 5, label = "model 2")
plt.errorbar(x = ensemble_sizes, y = median_log_like_list_3,
             fmt="o-", color="purple", ms = 5, label = "model 3")
plt.xlabel("Ensemble Size")
plt.ylabel("Median Log-Likelihood") # 10 iterations/instances
plt.legend()
plt.title("Median Log-Likelihood Against Ensemble Size")

plt.savefig(os.path.join(loc_save, "Plot_log_llhd_against_ensemble_size.png"), dpi=200)