import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import njit

@njit
def laplacian_numba(array, dx2):
    Nx, Ny = array.shape
    lap = np.zeros_like(array)
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            lap[i, j] = (array[i+1, j] + array[i-1, j] + array[i, j+1] + array[i, j-1] - 4 * array[i, j]) / dx2
    return lap

def time_dependent_2D_schrodinger(psi, potential, x, y, dx, dy, mass=1.0):
    hbar = 1.0  # natural units
    dx2 = dx**2
    # Use the JIT-compiled Laplacian
    lap = laplacian_numba(psi, dx2)
    rhs = -hbar**2/(2*mass) * lap + potential(x, y) * psi
    dpsi_dt = rhs / (hbar * 1j)
    return dpsi_dt

def wavefunction(x, y, t, sigma=1.0, k_x=1.0, k_y=1.0, omega=1.0, 
                 n=4, spacing_factor=3.0):
    # Calculate spacing between wave packet centers
    spacing = spacing_factor * sigma

    # Initialize the total wavefunction
    psi_total = np.zeros_like(x, dtype=complex)

    # Calculate offsets to center the grid
    offset = (n - 1) / 2 * spacing

    A = 1.0 / (n**2 * (2 * np.pi * sigma**2)**0.5)

    # Superpose Gaussian wave packets
    for i in range(n):
        for j in range(n):
            x0 = i * spacing - offset
            y0 = j * spacing - offset
            psi_total += A * np.exp(-(((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))) \
                                 * np.exp(1j*(k_x*x + k_y*y - omega*t))
    return psi_total
# Define the potential energy function
def potential(x, y):
    scale_field_1 = 30
    narrowness_field_1 = 10
    field_1 = narrowness_field_1 * ((x/scale_field_1)**2 + (y/scale_field_1)**2)
    
    
    scale_field_2 = 3e5
    U0 = 1.0
    r0 = 20.0
    r_squared = x**2 + y**2
    r_soft = np.sqrt(r_squared + r0**2)
    field_2 = U0 / (r_soft**3) * scale_field_2
        
    return field_1 + field_2
    
    


# Define grid
x = np.linspace(-50, 50, 500)
y = np.linspace(-50, 50, 500)
t = 0.0

X, Y = np.meshgrid(x, y)
dx = x[1] - x[0]
dy = y[1] - y[0]

# Initial wavefunction
psi = wavefunction(X, Y, t, sigma=2.0)

# Time evolution
schrodinger = psi.copy()
dt = 0.01  # Small time step
num_repetitions = 5
num_frames = 50000

# Set up plot
plt.figure(1, (5.0, 7.0))
plt.subplot(2, 1, 1)
plt.imshow(np.abs(psi)**2, cmap="bone")
plt.title("Original Probability Density")
plt.colorbar()

plt.subplot(2, 1, 2)
plt.imshow(potential(X, Y), cmap="bone")
plt.title("Potential Energy")
plt.colorbar()

fig, ax = plt.subplots(figsize=(8,6))
im = ax.imshow(np.abs(schrodinger)**2, cmap="bone")
ax.set_title("Probability Density |ψ|^2")
plt.colorbar(im, ax=ax)

def update(frame):
    global schrodinger
    
    for _ in range(num_repetitions):
        # One RK4 time evolution step
        k1 = dt * time_dependent_2D_schrodinger(schrodinger, potential, X, Y, dx, dy)
        k2 = dt * time_dependent_2D_schrodinger(schrodinger + k1/2, potential, X, Y, dx, dy)
        k3 = dt * time_dependent_2D_schrodinger(schrodinger + k2/2, potential, X, Y, dx, dy)
        k4 = dt * time_dependent_2D_schrodinger(schrodinger + k3, potential, X, Y, dx, dy)
        schrodinger += (k1 + 2*k2 + 2*k3 + k4) / 6

    # Optionally, enforce Dirichlet BC at boundaries (if desired): schrodinger[0,:] = schrodinger[-1,:] = schrodinger[:,0] = schrodinger[:,-1] = 0
    
    im.set_data(np.abs(schrodinger)**2)
    return [im]

ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=0, blit=True)

plt.tight_layout()
plt.show()