# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 19:19:03 2023

@author: Andras Poulsen
"""

import numpy as np                                     # Matlab like syntax for linear algebra and functions
import matplotlib as mpl
import matplotlib.pyplot as plt                        # Plots and figures like you know them from Matlab
from iminuit import Minuit                             # The actual fitting tool, better than scipy's
import sys                                             # Modules to see files and folders in directories
from scipy.optimize import curve_fit
import pandas as pd

#%%
sys.path.append('data')

def RMS_func(array):
    val = np.mean(array)
    var_tmp = 0
    for i in array:
        var_tmp =+ (i - val)**2
    sigma = (len(array)**(-1) * var_tmp)**(1/2)
    
    return np.array([val, sigma])

#%%


m = pd.read_csv('data/Appstat - Project - Measurements - Ball on incline.csv')
m = m.drop(columns=m.columns[0:1])
m = m.drop(columns=m.columns[6:])
m = m.drop([3])
m = m.transpose()
m = m.set_index([0])
headers = m.iloc[0]
m = m[1:]
m.columns = headers

for i in headers:
    m[i] = m[i].astype(float)


# %%% Calculating the positions of the sensors


d_ball = RMS_func(m['Ball diameter (mm)']) # mm
# d_ball[1] = d_ball[1] + 0.02 # Adding the error of the caliper (mm)

d_rail = RMS_func(m['Track width (cm)']*10) # Adding the error of the caliper (mm)
# d_rail[1] = d_rail[1] + 0.02 # mm

def Err_prop_pos(val1, val2):
    val = val1[0] - val2[0]
    
    sigma = np.sqrt((val1[0]*val2[1])**2 + (val1[1]*val2[0])**2)
    return np.array([val, sigma])

g1 = Err_prop_pos(RMS_func(m['Gate 1 start (cm)']),
              RMS_func(m['Gate 1 to middle (cm)']))
g2 = Err_prop_pos(RMS_func(m['Gate 2 start (cm)']),
              RMS_func(m['Gate 2 to middle (cm)']))
g3 = Err_prop_pos(RMS_func(m['Gate 3 start (cm)']),
              RMS_func(m['Gate 3 to middle (cm)']))
g4 = Err_prop_pos(RMS_func(m['Gate 4 start (cm)']),
              RMS_func(m['Gate 4 to middle (cm)']))
g5 = Err_prop_pos(RMS_func(m['Gate 5 start (cm)']),
              RMS_func(m['Gate 5 to middle (cm)']))

g_pos = np.array([g1,g2,g3,g4,g5])

# %% Calculating the angle of the rail

# %%% Direct measurement Method 1 (Calculating each measurement and thereafter combining them)
def Err_prop_theta(*vals):
    val_temp = 0
    for i in range(0,len(vals)): # 
        val_temp += vals[i][0]
    val = val_temp/len(vals)
    
    sigma_temp = 0
    for i in range(len(vals)):
        sigma_temp += ( (1/len(vals)) * vals[i][1] )**2
    sigma = np.sqrt(sigma_temp)
    return np.array([val, sigma])


theta1_fw = 90 - m['Theta (1)']
theta2_fw = m['Theta (2)'] - 90
theta1_rev = 90 - m['Theta reverse (1)']
theta2_rev = m['Theta reverse (2)'] - 90

theta1 = Err_prop_theta(RMS_func(theta1_fw),
                          RMS_func(theta2_fw),
                          RMS_func(theta1_rev),
                          RMS_func(theta2_rev)
                          )
# theta1[1] += 0.5 # Adding the estimated error of the protractor (deg)

# %%% Direct measurement Method 2 (Using all data points in one go)

theta_temp = pd.concat([theta1_fw,theta2_fw,theta1_rev,theta2_rev])
theta2 = RMS_func(theta_temp)
# theta2[1] += 0.5 # Adding the estimated error of the protractor (deg)

# %%% Calculating the angle with sinus

len_a = RMS_func(m['Theta Pyth Length'])
len_b = RMS_func(m['Theta Pyth Height'])

def theta_func(a,b):
    
    c2 = (a[0]**2 + b[0]**2)
    
    val = np.arcsin(b[0]/np.sqrt(c2))
    
    sigma = np.sqrt( ( b[0]**2 / (c2**(3/2) * np.sqrt(1 - a[0]**2/c2)) * a[1] )**2 + (a[0]**2 / ((c2**(3/2) * np.sqrt(1 - b[0]**2/c2)) * b[1]) )**2 )
    return np.array([val,sigma])*180/np.pi
    
theta_calc = theta_func(len_a, len_b)