import numpy as np 
import matplotlib.pyplot as plt

datapoints = 100
elements = 5
a = np.random.rand(datapoints,elements) * 10.0
for i in range(datapoints):
	for j in range(elements):
		if j == 1:
			a[i,j] = np.random.normal(0,100)

		if j == 2:
			a[i,j] = np.random.normal(0,20)

a = a[a[:,1].argsort()]

x = a[np.where(-1.0 < a[:,1] < -0.8)]

print(x)

plt.hist(force1,bins=5)
plt.show()
