#RFP Appendix 1 Part 2. Deriving longitude band of temperatures.

import numpy as np

#Question 1
T_terminator = 110 #K
T_optical = 293.15 #K

def temperature(phi, r):
    if r >= 0.3075 and r <=0.4667: #Mercury-Sun distance in AU, [r_min, r_max]
        if phi <= 90:
            T_subsolar = 407 + (8/np.sqrt(r))
            T = T_subsolar*(np.cos((np.deg2rad(phi))))**(1/4) + T_terminator*(phi/90)**3
        elif phi > 90:
            T = T_terminator
    return T

p = [89.1, 89.2, 89.3, 89.4, 89.5, 89.6, 89.7, 89.8, 89.9]
for i in p:
    #x = temperature(i, 0.4667)
    x = temperature(i, 0.3075)
    print(i, x)

#tasks: 
# 1. calculate lagging and advancing distances as angles of longitude. <done this, need to run by team
# 2. vary it over mercury's orbit
# 3. create a lookup table of surface temp vs lag & adv latitude relative to 20C