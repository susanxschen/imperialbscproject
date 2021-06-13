"""
this script creates a new std file for every single run of the model
- it puts a lower cap of 1e-2 on each std value to avoid variations between
  the ensemble size trends for noisy and noiseless data

run this before generating errors if required 
"""
import numpy as np

ensembles_list = [3,5,7,10,12,15,18,20]

for ensemble_size in ensembles_list:
    for iteration_num in range(1, 11):
        
        loc = f"main_folder_location/{ensemble_size} models/{iteration_num} iterations"
        
        std_old = np.loadtxt(f"{loc}/std.txt")
        num_values = np.shape(std_old)[0]
        
        std_new = []
        for i in range(num_values):
            if std_old[i] >= 3e-2:
                std_new.append(std_old[i])
            else:
                std_new.append(3e-2)
        
        std_new = np.array(std_new).reshape(-1,1)
        
        np.savetxt(f"{loc}/std_capped.txt", std_new)
        
