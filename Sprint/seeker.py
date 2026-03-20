import numpy as np
import scipy.integrate as integrate

# ====================================
#              Seeker
# ====================================

sigma = 0               # line-of-sight vector from seeker to tracking point (m)

d1 = 0                  # distance vector from missile center of mass to seeker (m)

d2 = 0                  # distance vector from target center of mass to tracking point

R = 0                   # range vector from missile center of mass to target center of mass, m

def seeker():
    sigma =