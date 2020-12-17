# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 10:21:22 2020

@author: Daan
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd

filename = 'Essen 20200813 1200.txt'
variables = ['PRES', 'HGHT', 'TEMP', 'DWPT', 'RELH', 'MIXR', 'DRCT', 'SKNT', 'THTA', 'THTE', 'THTV']
obs = pd.DataFrame(data = np.genfromtxt(filename), columns = variables).dropna()
del filename, variables

def T_env(z, obs = obs):
    """ Returns the environmental temperature given a specific height """
    
    h = obs['HGHT']
    T_atm = obs['TEMP']
            
    f = interp1d(h, T_atm, fill_value = 'extrapolate')
    
    return f(z)

def P_env(z, obs = obs):
    """ Returns the environmental temperature given a specific height """

    h = obs['HGHT']
    P_atm = obs['PRES'] * 100
           
    f = interp1d(h, P_atm, fill_value = 'extrapolate')
    
    return f(z)

def wv_sat(T, z):
    """ Computes the equilibrium water vapour pressure for a given temperature """

    Ra = 287.05
    Rv = 461.51
    eps = Ra/Rv  
    
    P_sat = 611.2 * np.exp(17.67 * T / (T + 243.5))
    
    return eps*P_sat/P_env(z)

def P_sat(T):
    return 611.2 * np.exp(17.67 * T / (T + 243.5))

def dw(T, z, wv, wl): 
    """ Updates the liquid water and water vapour mixing ratios """
    tau = 5
    delay = dt/tau  
    
    dwv = 0
    wvs = wv_sat(T,z)
    
    if wv > wvs:
        dwv = wv - wvs 
        dwv = delay*dwv
        wv = wv - dwv
        wl = wl + dwv
        
    elif wv < wvs and wl > 0:
        tau = 5
        delay = dt/tau
        dwv = wv - wvs
        dwv = delay*dwv
        
        if abs(dwv) > wl:
            dwv = -wl
            wv = wv - dwv
            wl = 0   
        else: 
            wv = wv - dwv
            wl = wl + dwv 
            
                    
    return dwv, wv, wl, wvs

def T_dew(T, wv, wvs):
    """ Calculate the dewpoint temperature from the RH and temperature """     
    
    return T - (100 - wv/wvs*100)/5
    

#=====================================================================================================
#============================================= MAIN CODE =============================================
#=====================================================================================================

# constants
g = 9.8         #  m/s2
gamma = 0.62    # 'frictional' term to take into account that an ascending air parcel first has to move air that is already in place. More strictly: m'/2m
Le = 2.5e6      # specific latent heat of evaporation for water. 
cpa = 1004.0    # J/kg

# things to choose
tmax = 2400          # integration time in seconds
dt = 0.1              # time step in seconds

# create arrays for value storage
time = np.arange(0,tmax,dt)

W = np.zeros_like(time)         # parcel velocity (m/s)
T_parc = np.zeros_like(time)    # parcel temperature (Celsius)
wv_parc = np.zeros_like(time)   # saturation mixing ratio
wl_parc = np.zeros_like(time)   # saturation mixing ratio
z = np.zeros_like(time)         # height (m)

# initial conditions
def buoyancy(T_parc, z):
    return 0*((T_parc+273.15)-(T_env(z)+273.15))/(T_env(z)+273.15)

def detr(T_parc, z):
    return -buoyancy(T_parc, z)


mu = 10**-4                     # entrainment rate, equals dln(m')
W[0] = 0.0                      # initial velocity of parcel
wv_parc[0] = 0.0134             # initial water vapour mixing ratio of parcel.
wl_parc[0] = 0.0                # initial liquid water mixing ratio of parcel.
wv_env = 0.0134                 # water vapour mixing ratio of environment.
wl_env = 0                      # liquid water mixing ratio of environment.
z[0] = 147                      # initial height of parcel
T_parc[0] = T_env(z[0]) + 0.1   # initial temperature parcel


# adjust wv and wl in case saturation already occures at the ground
dwv, wv_parc[0], wl_parc[0], wvsat = dw(T_parc[0], z[0], wv_parc[0], wl_parc[0])

# Euler forward scheme
for t in range(len(time)-1):
        
# update the velocity of the air parcel 
    wl_parc[t] = wl_parc[t] + mu*wl_env*dt - detr(T_parc[t], z[t])*wl_env*dt
    W[t+1] = W[t] + 1/(1+gamma)*g*dt*  (((T_parc[t]+273.15) - (T_env(z[t])+273.15))/(T_env(z[t])+273.15) - wl_parc[t] - mu*W[t] + detr(T_parc[t], z[t]))  
# compute the new height of the air parcel
    z[t+1] = z[t] + W[t+1]*dt

# compute the water vapour/liquid mixing ratio
    dwv, wv_parc[t+1], wl_parc[t+1], wvsat = dw(T_parc[t], z[t+1], wv_parc[t], wl_parc[t])
        
# update the temperature of the air parcel
    T_parc[t+1] = T_parc[t] - g/cpa*W[t+1]*dt + ( 
         Le/cpa*(dwv + mu*(wv_parc[t+1] - wv_env) - detr(T_parc[t],z[t+1])*(wv_parc[t+1] - wv_env))
         - mu*(T_parc[t] - T_env(z[t+1])) + detr(T_parc[t],z[t+1])*(T_parc[t] - T_env(z[t+1])))
    
    

#%%
# plotting
def Skew_T_diagram(T, z, wv): 
    """ Plot a Skew-T diagram. Credits @ MetPy: https://unidata.github.io/MetPy/latest/index.html """    

    from metpy.plots import SkewT
    
    P = P_env(z)/100
    wvs = wv_sat(T,z)     
    Td = T_dew(T, wv, wvs)
    
    env = T_env(z)  
    
    fig = plt.figure(figsize=(11,11))
    skew = SkewT(fig)
    skew.plot(P[0:3200], Td[0:3200], 'b',linewidth=3, label = 'T_dew')      # plot Tdew alleen tot LCL 
    skew.plot(P, T, 'r',linewidth=3, label = 'T_parc')
    skew.plot(P, env,'g', linewidth=3, label = 'T_env')
    skew.plot_dry_adiabats()
    skew.plot_moist_adiabats()
    skew.ax.set_ylim(1010,100)
    plt.legend()
    
    # Good bounds for aspect ratio
    skew.ax.set_xlim(-45, 50)
    skew.ax.set_xlabel('Temperature [C]')
    skew.ax.set_ylabel('Pressure [hPa]')
        
    return plt.show()

    
fig, ax = plt.subplots(3,1)

ax[0].plot(time, z)
ax[0].set_ylabel('height in m')
plt.setp(ax[0].get_xticklabels(), visible=False)

ax[1].plot(time, T_parc)
ax[1].set_ylabel('parcel temperature in C')
plt.setp(ax[1].get_xticklabels(), visible=False)

ax[2].plot(time, W)
ax[2].set_ylabel('W-velocity in m/s')
ax[2].set_xlabel('time in s')

# plot Skew-T diagram    
Skew_T_diagram(T_parc, z, wv_parc)


plt.figure()
plt.plot(time,wl_parc,label='Liquid water mixing ratio')
plt.plot(time,wv_parc,label='Vapor water mixing ratio')
plt.legend()
plt.show()

T_pr = T_env(z)
plt.figure()
plt.plot(time,T_parc,label='T parcel')
plt.plot(time,T_pr,label='T env')
plt.legend()
plt.show()