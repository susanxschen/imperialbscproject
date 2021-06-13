"""
testing the validity of the two_dimension_function.py code against analytical solutions
"""

import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import library
import two_dimensional_function as two_d

# analytical functions 

sigma_xy = lambda theta_1, theta_2: 0.5*(np.cos(theta_2) - np.sin(theta_2))**2
sigma_yz = lambda theta_1, theta_2: 0.5*(np.cos(theta_1) - np.sin(theta_1))**2
sigma_yx = lambda theta_1, theta_2: 0.5*((np.cos(theta_2)**2)*(1-2*np.cos(theta_1)*np.sin(theta_1)) 
                                        + (np.sin(theta_2)**2)*(1+2*np.cos(theta_1)*np.sin(theta_1)))
sigma_xz = lambda theta_1, theta_2: 0.5


theta_1_list = np.linspace(0,4,100)
theta_2_list = np.linspace(0,4,100)


z_analytical = []
for i in range(len(theta_2_list)):
    row=[]
    for j in range(len(theta_1_list)):
        F = sigma_xy(theta_1_list[j],theta_2_list[i])
        row.append(F)
    z_analytical.append(row)

#z_analytical = np.full((100,100), 0.5)

sns.set_style("whitegrid")

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(theta_1_list, theta_2_list, z_analytical, 100, cmap="viridis")
#test below
ax.contour3D(theta_1_list, theta_2_list, two_d.Z, 100, cmap="binary")
ax.set_xlabel(r'$\theta_{1}$')
ax.set_ylabel(r'$\theta_{2}$')
ax.set_zlabel(r"$|F(\theta_{1}, \theta_{2})|^{2}$")
ax.xaxis.labelpad=10
ax.yaxis.labelpad=10
ax.zaxis.labelpad=10


deviation_z = []
for i in range(100):
    row = []
    for j in range(100):
        deviation_ij = z_analytical[i][j] - two_d.Z[i][j]
        row.append(deviation_ij)
    deviation_z.append(row)
    
# if all elements in deviation_z is less than 10^-15 then good
for i in range(100):
    for j in range(100):
        if deviation_z[i][j] < 10e-15:
            pass
        else:
            print("Error in coded function!") 
            break
            

        







