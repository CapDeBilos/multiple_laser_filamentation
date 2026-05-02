import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, hermite

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



# Hermite not working for some reason. Will correct tomorrow

def HG(x, y, Pin=1, width=0.5, m=0, n=0):
    Hm = hermite(m)
    Hn = hermite(n)
    # normalized constant
    A = np.sqrt(Pin / (np.pi * w**2 * 2**(m + n - 1) * factorial(m) * factorial(n)))
    field = (A
             * Hm(np.sqrt(2) * x / w)
             * Hn(np.sqrt(2) * y / w)
             * np.exp(-(x**2 + y**2) / w**2))

    return field



x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)

E = HG(X, Y, Pin=1.0, w=0.5, m=1, n=2)
power = np.sum(np.abs(E)**2) * dx * dy
print(f"Integrated power: {power:.6f}")  # should be ~1.0

# Evaluate function
Z = HG(X, Y, Pin=1.0, w=0.5, m=1, n=2)
I = np.abs(Z)**2  # intensity

# Plot
plt.figure(figsize=(6, 5))
plt.imshow(I, extent=[x.min(), x.max(), y.min(), y.max()],
           origin='lower', cmap='inferno')
plt.colorbar(label='Intensity')
plt.title('HG mode (1,2)')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()