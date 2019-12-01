import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#plt.gca().spines['top'].set_visible(False)
# gridspec.GridSpec(3,3)
# plt.subplot2grid((2,3),(0,1)); plt.subplot2grid((2,3),(1,0),colspan=3)

save = True

setpoint_times = [10-2,60-2,120-2,140-2,190-2]

'''
Positional data
'''

posdata = np.genfromtxt('bagfile__observer_eta_ned.csv',delimiter=',')

north = posdata[1:,1:2]
east = posdata[1:,2:3]
psi = posdata[1:,6:7]
time = posdata[1:,7:]

# Points for the different box test square
squarex = [east[1,0],east[1,0],east[1,0] -5.0, east[1,0] -5.0, east[1,0]]
squarey = [north[1,0],north[1,0]+5,north[1,0] + 5.0, north[1,0], north[1,0]]

# NEDPOS
plt.figure()
plt.plot(east,north,color = '#741b47')
plt.scatter(squarex,squarey,color = '#bf9000',marker='8',s=40)
plt.xlabel('East [m]')
plt.ylabel('North [m]')

if save:
	plt.tight_layout()
	plt.savefig('nn_box_ned.pdf')

# HEADING
plt.figure()
plt.plot(time,psi,color = '#0b5394')
for c in setpoint_times:
	plt.axvline(c,ls='--',color='#bf9000')

plt.xlabel('Time [s]')
plt.ylabel('Yaw angle [deg]')

if save:
	plt.tight_layout()
	plt.savefig('nn_box_yaw.pdf')

'''
TAU DIFFERENCE
'''

taudata = np.genfromtxt('bagfile__NN_tau_diff.csv',delimiter=',')

taux = taudata[1:,1:2]
tauy = taudata[1:,2:3]
taup = taudata[1:,6:7]
time = taudata[1:,7:8]

plt.figure()
plt.subplot(311)
plt.plot(time,taux,color = '#b45f06')
for c in setpoint_times:
	plt.axvline(c,ls='--',color='#bf9000')
plt.ylabel('Force in surge [N]')
plt.subplot(312)
plt.plot(time,tauy,color = '#85200c')
for c in setpoint_times:
	plt.axvline(c,ls='--',color='#bf9000')
plt.ylabel('Force in sway [N]')
plt.subplot(313)
plt.plot(time,taup,color = '#38761d')
for c in setpoint_times:
	plt.axvline(c,ls='--',color='#bf9000')
plt.ylabel('Moment in yaw [Nm]')
plt.xlabel('Time [s]')

if save:
	plt.tight_layout()
	plt.savefig('nn_box_taudiff.pdf')

'''
Thruster outputs
'''

forces = np.genfromtxt('bagfile__NN_F.csv',delimiter=',')

T1 = forces[1:,1:2]
T2 = forces[1:,2:3]
T3 = forces[1:,3:4]

T1avg = np.mean(np.abs(T1))
T2avg = np.mean(np.abs(T2))
T3avg = np.mean(np.abs(T3))
averages = [T1avg,T2avg,T3avg]
colors = ['#38761d','#85200c','#b45f06']

objects = ('Port average thrust','Starboard average thrust','Bow average thrust')
y_pos = np.arange(len(objects))

plt.figure()
plt.bar(y_pos, averages, align='center', alpha=0.8,color=colors)
plt.xticks(y_pos, objects)
plt.ylabel('Force [N]')

if save:
	plt.tight_layout()
	plt.savefig('nn_box_avgForce.pdf')

plt.show() if not save else None
