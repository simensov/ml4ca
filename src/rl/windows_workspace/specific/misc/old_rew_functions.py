'''
This file just stores old tested reward functions
'''

def vel_reward_2(self, coeffs = None):
    ''' Penalizes high velocities as a sum instead of sqrt'''
    if coeffs is None:
        coeffs = self.vel_rew_coeffs
    assert len(coeffs) == 3 and isinstance(coeffs,list), 'Vel coeffs must be a list of length 3'

    pen = 0
    bnds = self.real_ss_bounds[-3:]
    vels = get_vel_3DOF(self.dTwin)
    for vel, bnd, cof in zip(vels,bnds,coeffs):
        pen -= np.abs(vel) / bnd * cof

    return pen

def multivar(self):
    surge, sway, yaw = self.EF.get_pose()
    r = np.sqrt(surge**2 + sway**2)
    yaw_vs_r_factor = 0.25 # how much one degree is weighted vs a meter
    r3d = np.sqrt(r**2 + (yaw * 180 / np.pi * 0.25)**2)
    return gaussian_like(val = [r3d], mean = [0], var = [2.0**2]) + max(0.0, (1-0.05*r3d))

def summed_gaussian_like(self):
    ''' First reward function that fixed the steady state error in yaw by sharpening the yaw gaussian '''
    surge, sway, yaw = self.EF.get_pose()
    rews = gaussian_like([surge,sway]) # mean 0 and var == 1
    yawrew = gaussian_like([yaw], var=[0.1**2]) # Before, using var = 1, there wasnt any real difference between surge and sway and yaw
    r = np.sqrt(surge**2 + sway**2) # meters
    special_measurement = np.sqrt(r**2 + (yaw * 0.25)**2) 
    anti_sparity = max(-1.0, (1-0.1*special_measurement))
    return sum(rews) / 2 + 2 * yawrew + anti_sparity

def summed_gaussian_with_multivariate(self):
    ''' First reward function that fixed the steady state error in yaw by sharpening the yaw gaussian '''
    surge, sway, yaw = self.EF.get_pose()

    yaw = yaw * 180 / np.pi # Use degrees since that was the standard when creating the reward function - easier visualized than radians
    yawrew = gaussian_like([yaw], var=[5.0**2]) # Before, using var = 1, there wasnt any real difference between surge and sway and yaw
    
    r = np.sqrt(surge**2 + sway**2) # meters
    radrew = gaussian_like([r]) # mean 0 and var 1

    special_measurement = np.sqrt(r**2 + (yaw * 0.25)**2) 
    anti_sparity = max(-1.0, (1-0.1*special_measurement))

    x = np.array([[r, yaw]]).T
    multi_rew = np.exp(-0.5 * (x.T).dot(self.covar_inv).dot(x))
    
    return radrew + yawrew + anti_sparity + multi_rew