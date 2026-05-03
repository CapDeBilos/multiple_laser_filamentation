import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, hermite

### Constants
epsilon0 = 8.854e-12 # F/m
c = 2.99793e8 # m/s
mu0 = 4 * np.pi * 1e-7 # H/m


### Define initial beam shapes

# Given an initial power, what is the intensity distribution of the laser?
# For a radially symmetric beam, the amplitude of the gaussian is found by normalizing the intensity so that it adds up to the input power
def supergaussian(r, Pin=1, width=0.5, p=2):
    '''
    Pin - input power
    p - the order of the supergaussian, use n=2 and you get normal gaussian
    '''
    I0 = p / (2 * np.pi * width ** 2) * Pin * 1 / gamma(2 / p)
    return I0 * np.exp(- (np.abs(r) / width) ** p)

def gaussian(r, Pin=1, width=0.5):
    return supergaussian(r, Pin, width, p=2)


# Supergaussian for the field, not the intensity
def supergaussian2(r, Pin=1, width=0.5, p=2):
    '''
    Pin - input power
    p - the order of the supergaussian, use n=2 and you get normal gaussian
    '''
    E = np.sqrt(Pin * p * 2 ** (2/p) / (np.pi * c * epsilon0 * width**2 * gamma(2/p)))
    return E * np.exp(- (np.abs(r) / width) ** p)

def gaussian2(r, Pin=1, width=0.5):
    return supergaussian2(r, Pin, width, p=2)

x = np.linspace(-1.5, 1.5, 1000)
plt.figure()
plt.plot(x, supergaussian2(x, p=20))
plt.show()


# Hermite-Gauss TEM modes (eigenmodes), normalized using the envelope of the field
def HG(x, y, Pin=1, width=0.5, l=0, m=0):
    Hm = hermite(l)
    Hn = hermite(m)
    A = np.sqrt(2 * Pin / (c * epsilon0 * np.pi * width**2 * 2**(l + m) * math.factorial(int(l)) * math.factorial(int(m)))) # normalized constant
    field = (A 
             * np.abs(Hm(np.sqrt(2) * x / width) * Hn(np.sqrt(2) * y / width))
             * np.exp(-(x**2 + y**2) / width**2))
    return field

# Superposition of TEM modes
'''
To follow up...
'''

# Add noise to a signal
'''
Coming soon...
'''



### Testing (veeeeeeeeeeeeeery nice picture)
'''
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
l, m = 3, 2
E = HG(X, Y, Pin=1.0, width=0.5, l=l, m=m)
plt.figure(figsize=(6, 5))
plt.imshow(E, extent=[x.min(), x.max(), y.min(), y.max()],
           origin='lower', cmap='inferno')
plt.colorbar(label='Field')
plt.title(f'HG mode {l, m}')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()
'''