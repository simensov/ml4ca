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