"""
regular data in while loop - for the ensembles analysis repititions
generates new seed data every repetition
"""
import numpy as np
import sys
import shutil

sys.path.append('../')
import library
import os

location = 'main_folder_location'

n_iterations = 1
while n_iterations <= 3:

    loc = os.path.join(location, f"repetition {n_iterations}/")
    shutil.rmtree(loc, ignore_errors = True)
    
    os.mkdir(loc)
    

    def main():
        #errors are saved to a .txt file
   #     library.error_handling(sys.argv[0][:-3])
        
        #save command line arguments to variables
        args = sys.argv
        
        print('these are the args: ',args)
        
        try:
            tm_type = args[1]
            n_samples = int(args[2])
            noise = args[3:]
        except IndexError:
            tm_type = "single_qubit"
            n_samples = 300
            noise = "g", 0, 0.05
          
        random = np.random.randint(1,1e5)
        
        #assign data path
        data_path = library.data_pathways_make_directories_regular(loc,tm_type, noise)
    

    
        #import range of theta from the library
        theta_min = library.theta_min
        theta_max = library.theta_max
        
    
        #create a toy model object with n_samples within our theta range
        tm = library.ToyModels(n_samples, theta_min, theta_max, seed=random)
        # here seed is changed every repetition
    
        if tm_type == 'cubic':
            x, y = tm.cubic(noise)
        elif tm_type == 'single_qubit':
            x, y = tm.single_qubit(noise)
        elif tm_type == 'single_qubit_even':
            x, y = tm.single_qubit_even(noise)
        elif tm_type == 'test_func_2':
            x, y = tm.test_func_2(noise)
        elif tm_type== 'landscape_test_1':
            x,y = tm.landscape_test_1(noise)    
        else:
            raise Exception('Unrecognised toy model.')
    
        #create a noise string
        noise_str = library.noise(noise)
    
        #save files in destinations
        np.savetxt(f'{data_path}/x-{n_samples}{noise_str}.csv', x,
                   delimiter=',')
        np.savetxt(f'{data_path}/y-{n_samples}{noise_str}.csv', y,
                   delimiter=',')
    
        """
        extract the theta values of these points and save them in a text file for error analysis
        """
        # order them
        x_ordered = sorted(x)
        
        path_theta = os.path.join(loc+"data", "theta_values.txt")
                
        np.savetxt(path_theta, x_ordered)
        
    if __name__ == '__main__':
        main()
        
    n_iterations += 1
