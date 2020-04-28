import numpy as np 
import matplotlib.pyplot as plt 

angles = np.linspace(-np.pi, np.pi, 11)
# sins = np.linspace(-1.0,1.0,5)
# coss = np.linspace(-1.0,1.0,5)

rad2deg = 180 / np.pi

sins = [np.sin(a) for a in angles]
coss = [np.cos(a) for a in angles]

for i, (s, c) in enumerate(zip(sins,coss)):
        print('Angle from sin and cos:')
        print('{:.2f}, {:.2f}'.format( (np.pi / 2 - np.arcsin(s)) * rad2deg  , (np.pi / 2 - np.arccos(c)) *rad2deg ))
        t = np.arctan2(s,c) * rad2deg
        print('Angle from arctan2:')
        print(t)
        print('Actual angle')
        print(angles[i] * rad2deg)

# for a in angles:
#     print(a * rad2deg)
#     s = np.sin(a)
#     c = np.cos(a)
#     t = np.arctan2(s,c)
#     print(t * rad2deg)
#     print('#')