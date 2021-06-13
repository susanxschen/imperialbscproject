
"""
regular data 2d
"""
import numpy as np
import sys
import shutil

sys.path.append('../')
import library_2d
import os


location = 'C:/Users/Zhiyi/Desktop/year_3_project/0315data'

n_iterations = 1
while n_iterations <= 3:
    

    loc = os.path.join(location, f"repetition {n_iterations}/")
    #shutil.rmtree(location, ignore_errors = True)
    
    #os.mkdir(location)
    

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
            tm_type = "qubit"
            n_samples = 300
            noise = "g", 0, 0.05
            
        
        #assign data path
        data_path = library_2d.data_pathways_make_directories_regular(loc,tm_type, noise)
    

    
        #import range of theta from the library
        theta_1_min = library_2d.theta_1_min
        theta_1_max = library_2d.theta_1_max
        theta_2_min = library_2d.theta_2_min
        theta_2_max = library_2d.theta_2_max
    
        #create a toy model object with n_samples within our theta range
        tm = library_2d.ToyModels(n_samples, theta_1_min, theta_1_max, theta_2_min, theta_2_max,seed=np.random.randint(1,1e5))
        # here seed is changed every repition
    
        if tm_type == 'qubit':
            x, y, z = tm.qubit(noise)
            
            # z is of shape (n_samples, n_samples), a matrix
            
# =============================================================================
#             print("z is",z)
#             print("shape of z ", np.shape(z))
# =============================================================================
  
        else:
            raise Exception('Unrecognised toy model.')
    
        #create a noise string
        noise_str = library_2d.noise(noise)
    
        #save files in destinations
        np.savetxt(f'{data_path}/x-{n_samples}{noise_str}.csv', x,
                   delimiter=',')
        np.savetxt(f'{data_path}/y-{n_samples}{noise_str}.csv', y,
                   delimiter=',')
        np.savetxt(f'{data_path}/z-{n_samples}{noise_str}.csv', z,
                   delimiter=',')
        
    
        """
        extract the theta values of these points and save them in a text file for error analysis
        """
        # order them
        x_ordered = sorted(x)
        y_ordered = sorted(y)
        
       
        path_theta_1 = os.path.join(loc+"data", "theta_1_values.txt")
        path_theta_2 = os.path.join(loc+"data", "theta_2_values.txt")
        
        
        np.savetxt(path_theta_1, x_ordered)
        np.savetxt(path_theta_2, y_ordered)
        
    if __name__ == '__main__':
        main()
        
    n_iterations += 1
