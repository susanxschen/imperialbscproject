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


# say this contains folders like "# models"
# within each one we'll have iteration #
# then the normal data, results folders 

# =============================================================================
# def error_on_mean(standard_dev, n):
#     
#     error_on_mean = standard_dev/np.sqrt(n)
#     
#     return error_on_mean
# 
# average_mean_dev_list = []
# average_abs_dev_list = []
# average_log_like_list = []
# ==========================================================================

models_names = ["Chris week 4", "leo 5% noise week 4", "Leo noiseless week 4",
                "Yifeng week 4"]

ensemble_sizes = [3, 5, 7, 10, 12, 15, 18, 20]

n = 10 # iteration number

median_mean_dev_list_chris = []
median_mean_dev_list_leo5 = []
median_mean_dev_list_leo_noiseless = []
median_mean_dev_list_yifeng = []

median_abs_dev_list_chris = []
median_abs_dev_list_leo5 = []
median_abs_dev_list_leo_noiseless = []
median_abs_dev_list_yifeng = []

median_log_like_list_chris =[]
median_log_like_list_leo5 =[]
median_log_like_list_leo_noiseless =[]
median_log_like_list_yifeng =[]

for name in models_names:
    
    loc = f'C:/Users/xious/Documents/Year 3/BSc Project/Quantifying accuracy/{name}'
    
    
    # =============================================================================
    # standard_error_on_average = []
    # abs_standard_error_on_average = []
    # log_standard_error_on_average = []
    # =============================================================================
    
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
            
            
    # =============================================================================
    #         average_rel_deviation_i = np.mean(rel_deviation_list_i)
    #         average_abs_deviation_i = np.mean(abs_deviation_list_i)
    #         average_log_i = np.mean(log_like_list_i)
    # =============================================================================
      
        median_rel_deviation_i = np.median(rel_deviation_list_i) 
        median_abs_deviation_i = np.median(abs_deviation_list_i)
        median_log_i = np.median(log_like_list_i)
            
        # appending mean, abs dev and log onto the main lists
        
        if name == "Chris week 4":
            median_mean_dev_list_chris.append(median_rel_deviation_i)
            median_abs_dev_list_chris.append(median_abs_deviation_i)
            median_log_like_list_chris.append(median_log_i)
            
        elif name == "Leo noiseless week 4":
            median_mean_dev_list_leo_noiseless.append(median_rel_deviation_i)
            median_abs_dev_list_leo_noiseless.append(median_abs_deviation_i)
            median_log_like_list_leo_noiseless.append(median_log_i)
            
        elif name == "Yifeng week 4":
            median_mean_dev_list_yifeng.append(median_rel_deviation_i)
            median_abs_dev_list_yifeng.append(median_abs_deviation_i)
            median_log_like_list_yifeng.append(median_log_i)
            
        else:
            median_mean_dev_list_leo5.append(median_rel_deviation_i)
            median_abs_dev_list_leo5.append(median_abs_deviation_i)
            median_log_like_list_leo5.append(median_log_i)
# =============================================================================
#     average_mean_dev_list.append(average_rel_deviation_i)
#     average_abs_dev_list.append(average_abs_deviation_i)
#     average_log_like_list.append(average_log_i)
#     
# =============================================================================
    # errorbar stuff for rel deviation
# =============================================================================
#     standard_dev_i = np.std(rel_deviation_list_i)  
#     standard_error_i = error_on_mean(standard_dev_i, n)
#     standard_error_on_average.append(standard_error_i)
# 
# =============================================================================
    
# =============================================================================
#     # errorbar stuff for abs deviation
#     std_abs_i = np.std(abs_deviation_list_i)  
#     ste_abs_i = error_on_mean(std_abs_i, n)
#     abs_standard_error_on_average.append(ste_abs_i)
# =============================================================================

    
# =============================================================================
#     # errorbar stuff for log 
#     std_log_i = np.std(log_like_list_i)  
#     ste_log_i = error_on_mean(std_log_i, n)
#     log_standard_error_on_average.append(ste_log_i)
# =============================================================================
    
loc_save = 'C:/Users/xious/Documents/Year 3/BSc Project/Quantifying accuracy/'

sns.set_style("darkgrid")

plt.figure()
plt.errorbar(x = ensemble_sizes, y = median_mean_dev_list_yifeng,
             fmt="o-", color="red", ms = 5, label = "Yifeng")
plt.errorbar(x = ensemble_sizes, y = median_mean_dev_list_chris,
             fmt="o-", color="blue", ms = 5, label = "Chris")
plt.errorbar(x = ensemble_sizes, y = median_mean_dev_list_leo5,
             fmt="o-", color="green", ms = 5, label = "Leo with noise")
plt.errorbar(x = ensemble_sizes, y = median_mean_dev_list_leo_noiseless,
             fmt="o-", color="purple", ms = 5, label = "Leo noiseless")
# =============================================================================
# plt.errorbar(x = ensemble_sizes, y = average_mean_dev_list, yerr = standard_error_on_average,
#              fmt=".-", color="blue", ms = 7, label = "Average")
# =============================================================================
plt.xlabel("Enesemble Size")
plt.ylabel("Median Mean Deviation (w.r.t. 1 std)")
plt.legend()
plt.title("Median Mean Deviation (w.r.t. 1 std) against Ensemble Size")
plt.savefig(os.path.join(loc_save, "Plot_rel_mean_dev_against_ensemble_size.png"), dpi=200)

plt.figure()
plt.errorbar(x = ensemble_sizes, y = median_abs_dev_list_yifeng,
             fmt="o-", color="red", ms = 5, label = "Yifeng")
plt.errorbar(x = ensemble_sizes, y = median_abs_dev_list_chris,
             fmt="o-", color="blue", ms = 5, label = "Chris")
plt.errorbar(x = ensemble_sizes, y = median_abs_dev_list_leo5,
             fmt="o-", color="green", ms = 5, label = "Leo with noise")
plt.errorbar(x = ensemble_sizes, y = median_abs_dev_list_leo_noiseless,
             fmt="o-", color="purple", ms = 5, label = "Leo noiseless")
# =============================================================================
# plt.errorbar(x = ensemble_sizes, y = average_abs_dev_list, yerr = standard_error_on_average,
#              fmt=".-", color="blue", ms = 7, label = "Average")
# =============================================================================
plt.xlabel("Enesemble Size")
plt.ylabel("Median Mean Deviation (Absolute)")
plt.legend()
plt.title("Median Mean Deviation (Absolute) against Ensemble Size")
plt.savefig(os.path.join(loc_save, "Plot_abs_mean_dev_against_ensemble_size.png"), dpi=200)

plt.figure()
# =============================================================================
# plt.errorbar(x = ensemble_sizes, y = average_log_like_list, yerr = log_standard_error_on_average,
#              fmt="o-", color="green", ms = 7, label = "Average")
# =============================================================================
plt.errorbar(x = ensemble_sizes, y = median_log_like_list_yifeng,
             fmt="o-", color="red", ms = 5, label = "Yifeng")
plt.errorbar(x = ensemble_sizes, y = median_log_like_list_chris,
             fmt="o-", color="blue", ms = 5, label = "Chris")
plt.errorbar(x = ensemble_sizes, y = median_log_like_list_leo5,
             fmt="o-", color="green", ms = 5, label = "Leo with noise")
plt.errorbar(x = ensemble_sizes, y = median_log_like_list_leo_noiseless,
             fmt="o-", color="purple", ms = 5, label = "Leo noiseless")
plt.xlabel("Ensemble Size")
plt.ylabel("Median Log-Likelihood") # 10 iterations/instances
plt.legend()
plt.title("Median Log-Likelihood Against Ensemble Size")

plt.savefig(os.path.join(loc_save, "Plot_log_llhd_against_ensemble_size.png"), dpi=200)