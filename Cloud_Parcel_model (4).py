# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 10:21:22 2020

@author: Daan
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd

def read_obs(filename):
    """ Reads the observational data into a pandas dataframe """
    
    variables = ['PRES', 'HGHT', 'TEMP', 'DWPT', 'RELH', 'MIXR', 'DRCT', 'SKNT', 'THTA', 'THTE', 'THTV']
    
    return pd.DataFrame(data = np.genfromtxt(filename), columns = variables).dropna()

# observations 
filename = 'Essen 20200813 1200.txt'
obs = read_obs(filename)

# =============================================================================
# def z_LCL(T, Td):
#     """ returns the approximate hight of the Lifted Condensation Level (LCL) """
#     
#     g = 9.8
#     Le = 2.5e6 
#     cpa = 1004.0
#     Ra = 287.05
#     Rv = 461.51
#     eps = Ra/Rv 
#     
#     Td_lapse = -g*(Td + 273)/(Le*eps)
#     T_dry_lapse = -g/cpa
#     
#     return (T - Td)/(Td_lapse - T_dry_lapse)
# =============================================================================

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

def dw(T, z, wv, wl, delay): 
    """ Updates the liquid water and water vapour mixing ratios """
    
    dwv = 0
    wvs = wv_sat(T,z)
    
    if wv > wvs:
        dwv = delay*(wv - wvs) 
        wv = wv - dwv
        wl = wl + dwv
        
    elif wv < wvs and wl > 0:
        
        dwv = delay*(wv - wvs)
       
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
    
def Skew_T_diagram(T, z, wv): 
    """ Plot a Skew-T diagram. Credits @ MetPy: https://unidata.github.io/MetPy/latest/index.html """    

    from metpy.plots import SkewT
    
    P = P_env(z)/100
    wvs = wv_sat(T,z)
     
    Td = T_dew(T, wv, wvs)
    env = T_env(z)     

    zLCL = np.argmin(abs(T-Td))
    P_LCL = P[:zLCL]
    Td = Td[:zLCL]
    
    fig = plt.figure(figsize=(11,11))
       
    skew = SkewT(fig, rotation=45)    
    skew.plot(P_LCL, Td, 'b',linewidth=3, label = 'T_dew')
    skew.plot(P, T, 'r',linewidth=3, label = 'T_parc')
    skew.plot(P, env,'g', linewidth=3, label = 'T_env')
    skew.plot_dry_adiabats()
    skew.plot_moist_adiabats()
    skew.plot_mixing_lines()
    skew.ax.set_ylim(1010,100)

    skew.plot(P[zLCL], T[zLCL], '.k', markersize=15, label = 'LCL = %.1f km' % np.round(z[zLCL]/1000,1))
    
    skew.shade_cin(P[zLCL:], env[zLCL:], T[zLCL:], label = 'CIN')
    skew.shade_cape(P[zLCL:], env[zLCL:], T[zLCL:], label = 'CAPE')
    
    plt.legend()
    
    skew.ax.set_xlim(-30, 40)
    skew.ax.set_xlabel('Temperature [C]')
    skew.ax.set_ylabel('Pressure [hPa]')
    skew.ax.set_title('Skew-T diagram Essen Sounding')
       
    return plt.show()

#=====================================================================================================
#============================================= MAIN CODE =============================================
#=====================================================================================================

# constants
g = 9.8         #  m/s2
tau = 5         # s, heuristic approach to model the time it takes to condensate/evaporate water
gamma = 0.62    # 'frictional' term to take into account that an ascending air parcel first has to move air that is already in place. More strictly: m'/2m
Le = 2.5e6      # J, specific latent heat of evaporation for water. 
cpa = 1004.0    # J/kg

# things to choose
tmax = 840            # integration time in seconds
dt = 0.1              # time step in seconds

# create arrays for value storage
time = np.arange(0,tmax,dt)

W = np.zeros_like(time)         # parcel velocity (m/s)
T_parc = np.zeros_like(time)    # parcel temperature (Celsius)
wv_parc = np.zeros_like(time)   # saturation mixing ratio
z = np.zeros_like(time)         # height (m)

# find the approximate height of the LCL
#LCL = z_LCL(obs['TEMP'][0], obs['DWPT'][0]) 

# initial conditions
#mu = 0                             # entrainment rate, equals dln(m')
W[0] = 1                          # initial velocity of parcel
wv_parc[0] = 0.0134                 # initial water vapour mixing ratio of parcel.
wl_parc = 0                         # initial liquid water mixing ratio of parcel.
wv_env = 0.0134                     # water vapour mixing ratio of environment.
wl_env = 0                          # liquid water mixing ratio of environment.
z[0] = obs['HGHT'][0]               # initial height of parcel
T_parc[0] = T_env(z[0]) + 0.1       # initial temperature parcel
delay = dt/tau                      # evaporation/condensation time

def B(T_parc,T_env):
    return g/(1+gamma)*(T_parc-T_env)/(T_env + 273.15)        # Buoyancy term 

def mu(W,B):
    C = 0.6
    a = 0.01                      # a term should be betweeen 0 and 1 according to paper 
    return C*a*B/W**2           # Entrainment according to Eqs. 2 from Chikira & Sugiyama (2010)

def mu2(W):
    mu_tau = 2.4*10**-3         # Entrainment accoring to Eqs.3 from Chikira & Sugiyama (2010)
    return mu_tau*1/W 

# adjust wv and wl in case saturation already occures at the ground
dwv, wv_parc[0], wl_parc, wvsat = dw(T_parc[0], z[0], wv_parc[0], wl_parc, delay)

# Euler forward scheme
for t in range(len(time)-1):
        
# update the velocity of the air parcel 
    wl_parc = wl_parc + mu(W[t],B(T_parc[t],T_env(z[t])))*wl_env*dt   
    W[t+1] = W[t] + 1/(1+gamma)*(g*dt*((T_parc[t] - T_env(z[t]))/(T_env(z[t]) + 273.15) - wl_parc - mu(W[t],B(T_parc[t],T_env(z[t])))*W[t]))

# compute the new height of the air parcel
    z[t+1] = z[t] + W[t+1]*dt

# compute the water vapour/liquid mixing ratio
    dwv, wv_parc[t+1], wl_parc, wvsat = dw(T_parc[t], z[t+1], wv_parc[t], wl_parc, delay)

# update the temperature of the air parcel
    T_parc[t+1] = T_parc[t] - g/cpa*W[t+1]*dt + Le/cpa*(dwv + mu(W[t],B(T_parc[t],T_env(z[t])))*(wv_parc[t+1] - wv_env)) - mu(W[t],B(T_parc[t],T_env(z[t])))*(T_parc[t] - T_env(z[t+1]))

# plotting
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

