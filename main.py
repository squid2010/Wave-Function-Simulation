import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from scipy.ndimage import convolve

def laplacian(array):
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])
    return convolve(array, kernel, mode='nearest')

def time_dependent_2D_schrodinger(psi, potential, x, y, dx, dy, mass=1.0):
    hbar = 1.0 # natural units

    rhs = -hbar**2/(2*mass) * laplacian(psi)/(dx**2) + potential(x, y) * psi
    
    dpsi_dt = rhs / (hbar * 1j)

    return dpsi_dt

def wavefunction(x, y, t):
    x_range = [-11, 9]
    y_range = [-4, 6]
    
    # Compute the base periodic function.
    base = np.cos((x+1)/7) * np.cos((y-1)/3) * np.exp(1j*t)
    # Create a mask that's True where (x,y) is inside the desired central region.
    mask = (x >= x_range[0]) & (x <= x_range[1]) & (y >= y_range[0]) & (y <= y_range[1])
    # Use np.where to set the function to 0 outside the central region.
    return np.where(mask, base, 0)
    
def potential(x, y):
    return np.where(np.logical_and(np.abs(x) < 2, np.abs(y) < 2), 0, 1000)


# Define grid
x = np.linspace(-20, 20, 100)
y = np.linspace(-20, 20, 100)
t = 0.0

X, Y = np.meshgrid(x, y)
dx = x[1] - x[0]
dy = y[1] - y[0]

# Initial wavefunction
psi = wavefunction(X, Y, t)

# Time evolution
schrodinger = psi.copy()
dt = 0.0000001  # Small time step
num_steps = 1000

for _ in range(num_steps):
    k1 = dt * time_dependent_2D_schrodinger(schrodinger, potential, X, Y, dx, dy)
    k2 = dt * time_dependent_2D_schrodinger(schrodinger + k1/2, potential, X, Y, dx, dy)
    k3 = dt * time_dependent_2D_schrodinger(schrodinger + k2/2, potential, X, Y, dx, dy)
    k4 = dt * time_dependent_2D_schrodinger(schrodinger + k3, potential, X, Y, dx, dy)
    
    schrodinger += (k1 + 2*k2 + 2*k3 + k4) / 6


# Plot initial and evolved wavefunctions (plotting probability densities)
plt.figure(1)
plt.imshow(np.abs(psi)**2, cmap='viridis')
plt.colorbar()
plt.title("Initial Wavefunction")

plt.figure(2)
plt.imshow(np.abs(schrodinger)**2, cmap='viridis')
plt.colorbar()
plt.title("Evolved Wavefunction")

plt.show()
