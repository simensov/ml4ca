import matplotlib.pyplot as plt 

x = [i if i > 0 else 0 for i in range(-20,20)]

plt.figure()
plt.plot(range(-20,20),x)
plt.xlabel('z')
plt.ylabel('f(z)')
plt.show()