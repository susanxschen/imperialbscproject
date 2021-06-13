"""
plots average mean deviation, median mean deviation, and median log likelihood against ensemble size
"""

import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sys.path.append('../')
import os
import re

loc = 'C:/Users/xious/Documents/Year 3/BSc Project/Quantifying accuracy/leo 5% noise week 4'
# say this contains folders like "# models"
# within each one we'll have iteration #
# then the normal data, results folders 

ensemble_sizes = [3, 5, 7, 10, 12, 15, 18, 20]

n = 10 # iteration number
def error_on_mean(standard_dev, n):
    
    error_on_mean = standard_dev/np.sqrt(n)
    
    return error_on_mean

average_mean_dev_list = []
average_abs_dev_list = []
average_log_like_list = []
median_mean_dev_list = []
median_abs_dev_list = []
median_log_like_list =[]

# =============================================================================
# ensemble_10_mean_dev =[]
# =============================================================================

standard_error_on_average = []
abs_standard_error_on_average = []
log_standard_error_on_average = []

# below i labels ensemble size, j labels iteration
for i in ensemble_sizes:
    
    # lists defined 
    rel_deviation_list_i = []
    abs_deviation_list_i = []
    log_like_list_i = []
    for j in range(1, 11):
        path_models_folder_ij = os.path.join(loc, f"repetition {j}/{i} models/")
        
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
        
        
        average_rel_deviation_i = np.mean(rel_deviation_list_i)
        average_abs_deviation_i = np.mean(abs_deviation_list_i)
        average_log_i = np.mean(log_like_list_i)
  
        median_rel_deviation_i = np.median(rel_deviation_list_i) 
        median_abs_deviation_i = np.median(abs_deviation_list_i)
        median_log_i = np.median(log_like_list_i)
        
# =============================================================================
#         # this is to plot a histogram test
#         if i == 10:
#             ensemble_10_mean_dev.append(rel_deviation_list_i)
#         else:
#             pass
# =============================================================================
        
    # appending mean, abs dev and log onto the main lists
    median_mean_dev_list.append(median_rel_deviation_i)
    median_abs_dev_list.append(median_abs_deviation_i)
    median_log_like_list.append(median_log_i)
    average_mean_dev_list.append(average_rel_deviation_i)
    average_abs_dev_list.append(average_abs_deviation_i)
    average_log_like_list.append(average_log_i)
    
    # errorbar stuff for rel deviation
    standard_dev_i = np.std(rel_deviation_list_i)  
    standard_error_i = error_on_mean(standard_dev_i, n)
    standard_error_on_average.append(standard_error_i)

    
    # errorbar stuff for abs deviation
    std_abs_i = np.std(abs_deviation_list_i)  
    ste_abs_i = error_on_mean(std_abs_i, n)
    abs_standard_error_on_average.append(ste_abs_i)

    
    # errorbar stuff for log 
    std_log_i = np.std(log_like_list_i)  
    ste_log_i = error_on_mean(std_log_i, n)
    log_standard_error_on_average.append(ste_log_i)
    
    

sns.set_style("darkgrid")

plt.figure()
plt.errorbar(x = ensemble_sizes, y = median_mean_dev_list,
             fmt="o-", color="red", ms = 7, label = "Median")
plt.errorbar(x = ensemble_sizes, y = average_mean_dev_list, yerr = standard_error_on_average,
             fmt=".-", color="blue", ms = 7, label = "Average")
plt.xlabel("Enesemble Size")
plt.ylabel("Mean Deviation (w.r.t. 1 std)")
plt.legend()
plt.title("Mean Deviation (w.r.t. 1 std) against Ensemble Size")
plt.savefig(os.path.join(loc, "Plot_rel_mean_dev_against_ensemble_size.png"), dpi=500)

plt.figure()
plt.errorbar(x = ensemble_sizes, y = median_abs_dev_list,
             fmt="o-", color="red", ms = 7, label = "Median")
plt.errorbar(x = ensemble_sizes, y = average_abs_dev_list, yerr = standard_error_on_average,
             fmt=".-", color="blue", ms = 7, label = "Average")
plt.xlabel("Enesemble Size")
plt.ylabel("Mean Deviation (Absolute)")
plt.legend()
plt.title("Mean Deviation (Absolute) against Ensemble Size")
plt.savefig(os.path.join(loc, "Plot_abs_mean_dev_against_ensemble_size.png"), dpi=500)

plt.figure()
plt.errorbar(x = ensemble_sizes, y = average_log_like_list, yerr = log_standard_error_on_average,
             fmt="o-", color="green", ms = 7, label = "Average")
plt.errorbar(x = ensemble_sizes, y = median_log_like_list,
             fmt="o-", color="purple", ms = 7, label = "Median")
plt.xlabel("Ensemble Size")
plt.ylabel("Log-Likelihood") # 10 iterations/instances
plt.legend()
plt.title("Log-Likelihood Against Ensemble Size")

plt.savefig(os.path.join(loc, "Plot_log_llhd_against_ensemble_size.png"), dpi=500)


# =============================================================================
# plt.figure()
# s = np.asarray(ensemble_10_mean_dev[0]).reshape(10,)
# plt.hist(s, bins=6) # lol it looks nothing like a gaussian which means the model is bad
# =============================================================================











