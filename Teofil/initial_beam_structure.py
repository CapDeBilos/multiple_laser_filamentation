import numpy as np
import matplotlib.pyplot as plt

### Define initial beam shapes

# Gaussian, y = a \exp(-(x-x0)^2/width^2)
def gaussian(x, amplitude=1, width=0.5, x0=0):
    return amplitude * np.exp(- (x - x0) ** 2 / width ** 2)

# The p parameter is the order of the supergaussian, p=1 is normal gaussian
def supergaussian(x, amplitude=1, width=0.5, x0=0, p=2):
    return amplitude * np.exp(- ((x - x0) ** 2 / width ** 2) ** p)



x = np.linspace(-2, 2, 100)
y = supergaussian(x)


plt.figure()
plt.plot(x, gaussian(x), label='p=1')
plt.plot(x, supergaussian(x, p=2), label='p=2')
plt.plot(x, supergaussian(x, p=3), label='p=3')
plt.legend()
plt.show()