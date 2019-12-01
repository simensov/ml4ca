import numpy as np 
import matplotlib.pyplot as plt 
import time

np.random.seed(int(time.time()))
x = []

a = np.linspace(-100,100,201)
b = np.linspace(-100,100,201)

ll = 1000
for _ in range(ll):
	i = np.random.choice(a)
	j = np.random.choice(b)
	x.append(i*j)


def chunks(l, n):
    n = max(1, n)
    return [l[i:i+n] for i in range(0, len(l), n)]


x.sort()
x = np.array(x)

dx = 500
ranges = []

# Increase occurences of few representations, and decrease occurences in many occurences
# Allow maximum five representations of elements within each 100th sector
for el in range(int(np.min(x)),int(np.max(x)),dx):

	# Find where x has elements within current range
	y = x[np.where((el < x) & (x < (el + dx)))]

	# If there was no occurences, there is nothing to add to the set: move on
	if y.shape[0] == 0:
		continue

	# If there are less than five occurences of datapoints in this range, add random datapoints from the set into the set
	# Store old array in order to avoid increasing the probability of the first chosen element getting picked several times
	ytemp = np.copy(y)
	while y.shape[0] < 5:
		y = np.hstack((y,np.random.choice(ytemp)))

	# If there exists more than five elements in this range, remove random elements until there are five left
	while y.shape[0] > 5:
		y = np.delete(y,np.random.randint(0,len(y)),0)

	# Extend the vector containing all datapoints
	for j in y:
		ranges.append(j)

plt.figure()
plt.subplot(211)
plt.hist(x,bins = int(ll / 100))
plt.subplot(212)
plt.hist(ranges,bins=len(ranges))


# Check how saturation can be done
thrust = np.array([[101,100,0,-101,2,-3]]).T
thrust[np.where(thrust > 100.0)] = 100.0
thrust[np.where(thrust < -100.0)] = -100.0
print(thrust)

# Check how elements can be easily removed
thrust = thrust[np.where(np.abs(thrust[:,:]) > 3)]
thrust = thrust.reshape((thrust.shape[0],1))
print(thrust)
# plt.show()