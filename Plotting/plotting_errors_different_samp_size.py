"""
plotting figures of merit against ensemble size for varying sample sizes 
"""
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sys.path.append('../')
import os
import re


sample_sizes = [10,20,50,100,200,300]

ensemble_sizes = [1,2,3, 5, 7, 10, 12, 15]

n = 3 # iteration number

median_mean_dev_list_10 = []
median_mean_dev_list_20 = []
median_mean_dev_list_50 = []
median_mean_dev_list_100 = []
median_mean_dev_list_200 = []
median_mean_dev_list_300 = []

median_abs_dev_list_10 = []
median_abs_dev_list_20 = []
median_abs_dev_list_50 = []
median_abs_dev_list_100 = []
median_abs_dev_list_200 = []
median_abs_dev_list_300 = []



median_log_like_list_10 =[]
median_log_like_list_20 =[]
median_log_like_list_50 =[]
median_log_like_list_100 =[]
median_log_like_list_200 =[]
median_log_like_list_300 =[]

for sample_num in sample_sizes:
    
    loc = f'main_folder_location/single qubit/{sample_num} samples'
    
    
    # below i labels ensemble size, j labels iteration
    for i in ensemble_sizes:
        
        # lists defined 
        rel_deviation_list_i = []
        abs_deviation_list_i = []
        log_like_list_i = []
        for j in range(1, 4):
            path_models_folder_ij = os.path.join(loc, f"repetition {j}/{i} models/")

            
            # list of the relevant measures for the 3 iteration values for one ensemble size i
            
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
            if log_like_ij == []:
                log_like_ij = [np.nan]
            else:
                log_like_ij = log_like_ij
            
            rel_deviation_list_i.append(rel_mean_deviation_ij)
            abs_deviation_list_i.append(abs_mean_deviation_ij)
            log_like_list_i.append(log_like_ij)
            
      
        median_rel_deviation_i = np.median(rel_deviation_list_i) 
        median_abs_deviation_i = np.median(abs_deviation_list_i)
        median_log_i = np.median(log_like_list_i)
            
        # appending mean, abs dev and log onto the main lists
        
        if sample_num == 10:
            median_mean_dev_list_10.append(median_rel_deviation_i)
            median_abs_dev_list_10.append(median_abs_deviation_i)
            median_log_like_list_10.append(median_log_i)       
            
        elif sample_num == 20:
            median_mean_dev_list_20.append(median_rel_deviation_i)
            median_abs_dev_list_20.append(median_abs_deviation_i)
            median_log_like_list_20.append(median_log_i)
            
        elif sample_num == 50:
            median_mean_dev_list_50.append(median_rel_deviation_i)
            median_abs_dev_list_50.append(median_abs_deviation_i)
            median_log_like_list_50.append(median_log_i)
            
        elif sample_num == 100:
            median_mean_dev_list_100.append(median_rel_deviation_i)
            median_abs_dev_list_100.append(median_abs_deviation_i)
            median_log_like_list_100.append(median_log_i)
            
        elif sample_num == 200:
            median_mean_dev_list_200.append(median_rel_deviation_i)
            median_abs_dev_list_200.append(median_abs_deviation_i)
            median_log_like_list_200.append(median_log_i)
            
        else:
            median_mean_dev_list_300.append(median_rel_deviation_i)
            median_abs_dev_list_300.append(median_abs_deviation_i)
            median_log_like_list_300.append(median_log_i)
            
# =============================================================================
# plotting
# =============================================================================
loc_save = 'C:/Users/xious/Documents/Year 3/BSc Project/Quantifying accuracy/single qubit/'

sns.set_style("darkgrid")

plt.figure()
plt.errorbar(x = ensemble_sizes, y = median_mean_dev_list_10,
             fmt="o-", color="red", ms = 3, label = "10 samples", linewidth=1)
plt.errorbar(x = ensemble_sizes, y = median_mean_dev_list_20,
             fmt="o-", color="blue", ms = 3, label = "20 samples", linewidth=1)
plt.errorbar(x = ensemble_sizes, y = median_mean_dev_list_50,
             fmt="o-", color="green", ms = 3, label = "50 samples", linewidth=1)
plt.errorbar(x = ensemble_sizes, y = median_mean_dev_list_100,
             fmt="o-", color="purple", ms = 3, label = "100 samples", linewidth=1)
plt.errorbar(x = ensemble_sizes, y = median_mean_dev_list_200,
             fmt="o-", color="orange", ms = 3, label = "200 samples", linewidth=1)
plt.errorbar(x = ensemble_sizes, y = median_mean_dev_list_300,
             fmt="o-", color="grey", ms = 3, label = "300 samples", linewidth=1)

plt.xlabel("Ensemble Size")
plt.ylabel("Log Median Mean Deviation (Relative to 1 std.)")
plt.yscale("log")
plt.legend()
plt.title("Median Mean Deviation against Ensemble Size (Relative to 1 std.)")
plt.savefig(os.path.join(loc_save, "50 Plot_rel_mean_dev_against_ensemble_size.png"), dpi=200)

plt.figure()
plt.errorbar(x = ensemble_sizes, y = median_abs_dev_list_10,
             fmt="o-", color="red", ms = 3, label = "10 samples", linewidth=1)
plt.errorbar(x = ensemble_sizes, y = median_abs_dev_list_20,
             fmt="o-", color="blue", ms = 3, label = "20 samples", linewidth=1)
plt.errorbar(x = ensemble_sizes, y = median_abs_dev_list_50,
             fmt="o-", color="green", ms = 3, label = "50 samples", linewidth=1)
plt.errorbar(x = ensemble_sizes, y = median_abs_dev_list_100,
             fmt="o-", color="purple", ms = 3, label = "100 samples", linewidth=1)
plt.errorbar(x = ensemble_sizes, y = median_abs_dev_list_200,
             fmt="o-", color="orange", ms = 3, label = "200 samples", linewidth=1)
plt.errorbar(x = ensemble_sizes, y = median_abs_dev_list_300,
             fmt="o-", color="grey", ms = 3, label = "300 samples", linewidth=1)


plt.xlabel("Ensemble Size")
plt.ylabel("Log Median Mean Deviation (Absolute)")
plt.yscale("log")
plt.legend(loc=0)
plt.title("Median Mean Deviation against Ensemble Size (Absolute)")
plt.savefig(os.path.join(loc_save, "100 Plot_abs_mean_dev_against_ensemble_size.png"), dpi=200)

plt.figure()
plt.errorbar(x = ensemble_sizes, y = median_log_like_list_10,
             fmt="o-", color="red", ms = 3, label = "10 samples", linewidth=1)
plt.errorbar(x = ensemble_sizes, y = median_log_like_list_20,
             fmt="o-", color="blue", ms = 3, label = "20 samples", linewidth=1)
plt.errorbar(x = ensemble_sizes, y = median_log_like_list_50,
             fmt="o-", color="green", ms = 3, label = "50 samples", linewidth=1)
plt.errorbar(x = ensemble_sizes, y = median_log_like_list_100,
             fmt="o-", color="purple", ms = 3, label = "100 samples", linewidth=1)
plt.errorbar(x = ensemble_sizes, y = median_log_like_list_200,
             fmt="o-", color="orange", ms = 3, label = "200 samples", linewidth=1)
plt.errorbar(x = ensemble_sizes, y = median_log_like_list_300,
             fmt="o-", color="grey", ms = 3, label = "300 samples", linewidth=1)
plt.xlabel("Ensemble Size")
plt.ylabel("Median Log-Likelihood") # 10 iterations/instances
plt.legend()
plt.title("Median Log-Likelihood Against Ensemble Size")

plt.savefig(os.path.join(loc_save, "100 Plot_log_llhd_against_ensemble_size.png"), dpi=200)

