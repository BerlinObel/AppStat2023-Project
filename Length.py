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
sys.path.append('Data')

def RMS_func(values, sigmas=None):
    if sigmas is None:
        val = np.mean(values)
        val_tmp = 0
        for i in values:
            val_tmp =+ (i - val)**2
        sigma = (len(values)**(-1) * val_tmp)**(1/2)
    else:
        val_tmp = 0
        sigma_tmp = 0
        for i in range(len(values)):
            val_tmp =+ (values[i]/sigmas[i]**2)
            sigma_tmp =+ 1/sigmas[i]**2
        val = val_tmp/sigma_tmp
        sigma = np.sqrt(1/sigma_tmp)
    return np.array([val, sigma])

# %% Reading data
# %%% Ball on incline
m_BOI = pd.read_csv('Data/ball_on_incline_measurements.csv')
m_BOI = m_BOI.drop(columns=m_BOI.columns[0:1])
m_BOI = m_BOI.drop(columns=m_BOI.columns[6:])
m_BOI = m_BOI.drop([3])
m_BOI = m_BOI.transpose()
m_BOI = m_BOI.set_index([0])
headers = m_BOI.iloc[0]
m_BOI = m_BOI[1:]
m_BOI.columns = headers

for i in headers:
    m_BOI[i] = m_BOI[i].astype(float)

# %%% Pendulum
m_P = pd.read_csv('Data/pendulum_measurements.csv')

m_P = m_P.drop(columns=m_P.columns[0:1])
m_P = m_P.drop(columns=m_P.columns[7:])
m_P = m_P.drop([0,10,11])
m_P = m_P.T
m_P = m_P.set_index([1])
headers = m_P.iloc[0]
m_P = m_P[1:]
m_P.columns = headers

for i in headers:
    m_P[i] = m_P[i].astype(float)

# %% Calculations for Pendulum

# %%% 

l_PendulumToHook_START = RMS_func(m_P['Length from top to pendulum (cm)'], 
                                m_P['Length from top to pendulum uncertainty (mm)']*0.1)
w_Pendulum = RMS_func(m_P['Height of pendulum (cm)'][:-1], 
                                  m_P['Height of pendulum uncertainty (mm)'][:-1]*0.1)
l_COM_START = np.array( [l_PendulumToHook_START[0]+1/2*w_Pendulum[0],
                         np.sqrt(l_PendulumToHook_START[1]**2 + 1/2*w_Pendulum[1]**2)] )


l_PendulumToFloor_START = RMS_func(m_P['Length from pendulum to floor (cm)'][:-1], 
                             m_P['Length from pendulum to floor uncertainty (mm)'][:-1]*0.1)
l_PendulumToFloor_END = RMS_func(m_P['End length from floor to pendulum (cm)'], 
                             m_P['End length from floor to pendulum uncertainty (mm)']*0.1)

l_PendulumToFloor_CHANGE = np.array([l_PendulumToFloor_START[0] - l_PendulumToFloor_END[0], 
                                    np.sqrt(l_PendulumToFloor_START[1]**2 + l_PendulumToFloor_END[1]**2)])

l_PendulumToHook_END = np.array([l_PendulumToHook_START[0] + l_PendulumToFloor_CHANGE[0], 
                                 np.sqrt(l_PendulumToHook_START[1]**2 + l_PendulumToFloor_CHANGE[1]**2)])
l_COM_END = np.array( [l_PendulumToHook_END[0]+1/2*w_Pendulum[0],
                         np.sqrt(l_PendulumToHook_END[1]**2 + 1/2*w_Pendulum[1]**2)] )





# %% Calculations for Ball on incline
# %%% Calculating the positions of the sensors

d_ball = RMS_func(m_BOI['Ball diameter (mm)']) # mm
# d_ball[1] = d_ball[1] + 0.02 # Adding the error of the caliper (mm)

d_rail = RMS_func(m_BOI['Track width (cm)']*10) # Adding the error of the caliper (mm)
# d_rail[1] = d_rail[1] + 0.02 # mm

def Err_prop_pos(val1, val2):
    val = val1[0] - val2[0]
    
    sigma = np.sqrt((val2[1])**2 + (val2[1])**2)
    return np.array([val, sigma])

g1 = Err_prop_pos(RMS_func(m_BOI['Gate 1 start (cm)']),
              RMS_func(m_BOI['Gate 1 to middle (cm)']))
g2 = Err_prop_pos(RMS_func(m_BOI['Gate 2 start (cm)']),
              RMS_func(m_BOI['Gate 2 to middle (cm)']))
g3 = Err_prop_pos(RMS_func(m_BOI['Gate 3 start (cm)']),
              RMS_func(m_BOI['Gate 3 to middle (cm)']))
g4 = Err_prop_pos(RMS_func(m_BOI['Gate 4 start (cm)']),
              RMS_func(m_BOI['Gate 4 to middle (cm)']))
g5 = Err_prop_pos(RMS_func(m_BOI['Gate 5 start (cm)']),
              RMS_func(m_BOI['Gate 5 to middle (cm)']))

g_pos = np.array([g1,g2,g3,g4,g5])

# %%% Calculating the angle of the rail

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


theta1_fw = 90 - m_BOI['Theta (1)']
theta2_fw = m_BOI['Theta (2)'] - 90
theta1_rev = 90 - m_BOI['Theta reverse (1)']
theta2_rev = m_BOI['Theta reverse (2)'] - 90

theta_sep = Err_prop_theta(RMS_func(theta1_fw),
                          RMS_func(theta2_fw),
                          RMS_func(theta1_rev),
                          RMS_func(theta2_rev)
                          )
# theta1[1] += 0.5 # Adding the estimated error of the protractor (deg)

# %%% Direct measurement Method 2 (Using all data points in one go)

theta_temp = pd.concat([theta1_fw,theta2_fw,theta1_rev,theta2_rev])
theta_all = RMS_func(theta_temp)
# theta2[1] += 0.5 # Adding the estimated error of the protractor (deg)

# %%% Calculating the angle with sinus

len_a = RMS_func(m_BOI['Theta Pyth Length'])
len_b = RMS_func(m_BOI['Theta Pyth Height'])

def theta_func(a,b):
    
    c2 = (a[0]**2 + b[0]**2)
    
    val = np.arcsin(b[0]/np.sqrt(c2))
    
    sigma = np.sqrt( ( b[0]**2 / (c2**(3/2) * np.sqrt(1 - a[0]**2/c2)) * a[1] )**2 + (a[0]**2 / ((c2**(3/2) * np.sqrt(1 - b[0]**2/c2)) * b[1]) )**2 )
    return np.array([val,sigma])*180/np.pi
    
theta_calc = theta_func(len_a, len_b)

# %% Exporting the calculated data
values_P = pd.DataFrame()
values_P['l_PendulumToHook_START'] = l_PendulumToHook_START
values_P['w_Pendulum'] = w_Pendulum
values_P['l_COM_START'] = l_COM_START

values_P['l_PendulumToFloor_START'] = l_PendulumToFloor_START
values_P['l_PendulumToFloor_END'] = l_PendulumToFloor_END
values_P['l_PendulumToFloor_CHANGE'] = l_PendulumToFloor_CHANGE
values_P['l_PendulumToHook_END'] = l_PendulumToHook_END
values_P['l_COM_END'] = l_COM_END


g_pos_df = pd.DataFrame(g_pos, index=['g1','g2','g3','g4','g5'])
values_BOI = pd.DataFrame()
values_BOI['d_rail'] = d_rail
values_BOI['d_ball'] = d_ball
values_BOI['len_a'] = len_a
values_BOI['len_b'] = len_b
values_BOI['theta_sep'] = theta_sep
values_BOI['theta_all'] = theta_all
values_BOI['theta_calc'] = theta_calc
values_BOI['pos1'] = g_pos[0]
values_BOI['pos2'] = g_pos[1]
values_BOI['pos3'] = g_pos[2]
values_BOI['pos4'] = g_pos[3]
values_BOI['pos5'] = g_pos[4]

m_BOI.to_pickle('./Data/measurements_ball_on_incline')
values_BOI.to_pickle('./Data/values_BOI')

m_P.to_pickle('./Data/measurements_pendulum')
values_P.to_pickle('./Data/values_P')




