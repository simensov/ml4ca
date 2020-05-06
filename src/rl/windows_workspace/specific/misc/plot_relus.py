import matplotlib.pyplot as plt 
from plot_commons import set_params

set_params()

vals = [i - 5 for i in range(11)]

def relu(val): return max(0.0, val)
def leaky_relu(val): return 0.05 * val if val < 0.0 else val

relus = [relu(val) for val in vals]
leaky_relus = [leaky_relu(val) for val in vals]

plot_vals = [relus,leaky_relus]
zeros = [0 for i in range(11)]

f, axes = plt.subplots(1,2, sharey=True)
axes[0].set_ylabel('f(z)')

for axn, ax in enumerate(axes):
    ax.set_xlabel('z')
    ax.plot(vals,zeros,'--',color='black', alpha=0.8)
    ax.plot(vals, plot_vals[axn])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

f.tight_layout()
plt.show()