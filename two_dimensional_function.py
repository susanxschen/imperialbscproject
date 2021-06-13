"""
2d Landscape 
"""
import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import library
# =============================================================================
# from scipy.linalg import expm, sinm, cosm
# =============================================================================

# kets are column vectors, bras are row vectors

zero_ket = np.array([[1],
                     [0]])
one_ket = np.array([[0],
                    [1]])

def state(a, b):
    
    general_state = a * zero_ket + b * one_ket
    
    return general_state

# defining the control landscape

def control_landscape(state, theta_1, theta_2):
    
    def expon_i(theta, spin_matrix):
        
        """
        expon_i = np.exp(complex(0, 1) * theta * spin_matrix)
        
        Below expression is equal to above expression is matrix^2 = unit matrix. 
        
        """
        expon_i = (np.cos(theta) * np.identity(2)) + (complex(0, 1) * np.sin(theta) * spin_matrix)
        
# =============================================================================
#         expon_i = expm(complex(0, 1)*theta*spin_matrix) also works 
# =============================================================================
        
        return expon_i

# =============================================================================
#     expon_i = lambda theta, matrix: (np.cos(theta) * np.identity(2)) + (complex(0, 1) * np.sin(theta) * matrix)
# =============================================================================
    
    spin_matrix_x = np.array([[0, 1],
                              [1, 0]])
    spin_matrix_y = complex(0, 1) * np.array([[0, -1],
                                              [1, 0]])
    spin_matrix_z = np.array([[1, 0],
                             [0, -1]])
    
    exp_prod = np.dot(expon_i(theta_1, spin_matrix_x) ,expon_i(theta_2, spin_matrix_y))
    
    state_2 = np.dot(exp_prod, zero_ket)
    
    # the vdot conjugates the first term
    inner_product = np.vdot(state, state_2)
    
    F = abs(inner_product) ** 2
    
    return F


# parameter inputs

a = 1/np.sqrt(2)
b = 1/np.sqrt(2)

psi_state = state(a, b)

# =============================================================================
# theta_1 = 0.5
# theta_2 = 0
#     
# F = control_landscape(psi_state, theta_1, theta_2)
# print("F(theta_1, theta_2):", F)
# =============================================================================

# plotting the landscape of the qubit state

theta_1_min=0
theta_1_max=np.pi
theta_2_min=0
theta_2_max=np.pi


theta_1_list=np.linspace(theta_1_min,theta_1_max,num=100)
theta_2_list=np.linspace(theta_2_min,theta_2_max,num=100)

# function values
Z = []

for i in range(len(theta_2_list)):
    row=[]
    for j in range(len(theta_1_list)):
        F = control_landscape(psi_state, theta_1_list[j],theta_2_list[i])
        row.append(F)
    Z.append(row)
    

sns.set_style("whitegrid")

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(theta_1_list,theta_2_list, Z, 150, cmap='viridis')
ax.set_xlabel(r'$\theta_{1}$')
ax.set_ylabel(r'$\theta_{2}$')
ax.set_zlabel(r"$F(\theta_{1}, \theta_{2})$")
plt.xticks([0, 1/4*np.pi, 1/2*np.pi, 3/4*np.pi, np.pi], [0, r'$\pi$/4', r'$\pi$/2', r'3$\pi$/4', r'$\pi$'])  # Set text labels.
plt.yticks([0, 1/4*np.pi, 1/2*np.pi, 3/4*np.pi, np.pi], [0, r'$\pi$/4', r'$\pi$/2', r'3$\pi$/4', r'$\pi$']) 
ax.xaxis.labelpad=20
ax.yaxis.labelpad=20
ax.zaxis.labelpad=20
plt.savefig("xy comb", dpi=500)



