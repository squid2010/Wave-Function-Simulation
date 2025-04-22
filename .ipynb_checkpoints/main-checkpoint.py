import sys
import time
import numpy as np
from numba import njit, prange
from scipy.special import hermite, factorial
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

# --- Simulation functions ---

@njit(parallel=True)
def laplacian(array, dx2):
    Nx, Ny, Nz = array.shape
    lap = np.zeros_like(array)
    for i in prange(1, Nx-1):
        for j in prange(1, Ny-1):
            for k in prange(1, Nz-1):
                lap[i, j, k] = (array[i+1, j, k] + array[i-1, j, k] +
                                array[i, j+1, k] + array[i, j-1, k] +
                                array[i, j, k+1] + array[i, j, k-1] -
                                6 * array[i, j, k]) / dx2
    return lap

def time_dependent_3D_schrodinger(psi, potential, x, y, z, dx, dy, dz, mass=1.0):
    hbar = 1.0  # natural units
    dx2 = dx**2
    lap = laplacian(psi, dx2)
    rhs = -hbar**2/(2*mass) * lap + potential(x, y, z) * psi
    return rhs / (hbar * 1j)

def wavefunction(x, y, z, t, a0=2):
    r = np.sqrt(x**2 + y**2 + z**2)
    return (1/np.sqrt(np.pi*a0**3)) * np.exp(-r / a0)

@njit(parallel=True)
def potential(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    epsilon = 1e-4  # small softening to avoid singularity at r=0
    return -1.0 / np.sqrt(r**2 + epsilon)

def crank_nicolson_step(psi, dt, tol=1e-6, max_iter=10):
    psi_next = psi.copy()
    for _ in range(max_iter):
        f_old = time_dependent_3D_schrodinger(psi, potential, X, Y, Z, dx, dy, dz)
        f_new = time_dependent_3D_schrodinger(psi_next, potential, X, Y, Z, dx, dy, dz)
        psi_new = psi + (dt/2) * (f_old + f_new)
        if np.linalg.norm(psi_new - psi_next) < tol:
            psi_next = psi_new
            break
        psi_next = psi_new
    return psi_next

# --- Color computation ---
def compute_colors(psi_magnitude, alpha_exp=2.0):
    prob_density = psi_magnitude**2
    max_val = np.max(prob_density)
    normed = prob_density / max_val if max_val > 0 else prob_density
    t = normed.ravel()
    lowColor = np.array([0.0, 0.0, 0.12])
    highColor = np.array([1.0, 0.95, 0.88])
    rgb = (1 - t)[:, None] * lowColor + t[:, None] * highColor
    alpha = (t ** alpha_exp)[:, None]
    colors = np.hstack((rgb, alpha)).astype(np.float32)
    return colors

# --- Grid Resizing ---
def compute_resolution(psi_magnitude, threshold=1e-4, max_resolution=128):
    """
    Dynamically computes the grid resolution based on the wavefunction's magnitude.
    The resolution will be higher where the wavefunction magnitude is large.
    
    :param psi_magnitude: Wavefunction array, absolute value
    :param threshold: Minimum magnitude to increase resolution
    :param max_resolution: Maximum resolution allowed for grid refinement
    :return: Grid resolution (Nx, Ny, Nz)
    """

    # Compute the local resolution based on the wavefunction's magnitude
    # Areas where |psi| > threshold will have higher resolution
    resolution = np.ones_like(psi_magnitude) * max_resolution
    resolution[psi_magnitude < threshold] = max_resolution // 2  # Coarser resolution in low magnitude areas
    
    return resolution
    
def create_adaptive_grid(psi_magnitude, base_resolution=65, bound=5.0):
    global X, Y, Z
    """
    Creates an adaptive grid for the wavefunction based on its magnitude.
    
    :param psi: The wavefunction array (current state).
    :param base_resolution: The base resolution (for regions where no refinement is needed).
    :param bound: The bounds for the grid.
    :return: Adaptive grid (X, Y, Z, and resolution).
    """
    resolution = compute_resolution(psi_magnitude)
    
    # Adaptive mesh based on the computed resolution
    x = np.linspace(-bound, bound, base_resolution)
    y = np.linspace(-bound, bound, base_resolution)
    z = np.linspace(-bound, bound, base_resolution)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Create new grid with the adaptive resolution
    Nx, Ny, Nz = X.shape
    X = np.zeros((Nx, Ny, Nz))
    Y = np.zeros((Nx, Ny, Nz))
    Z = np.zeros((Nx, Ny, Nz))
    
    # Dynamically adjust grid spacing based on resolution
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                res = resolution[i, j, k]
                X[i, j, k] = np.linspace(-bound, bound, res)
                Y[i, j, k] = np.linspace(-bound, bound, res)
                Z[i, j, k] = np.linspace(-bound, bound, res)
    
    return X, Y, Z, resolution


# --- Set up the grid and initial conditions ---
Nx = Ny = Nz = 65
bound = 5
x = np.linspace(-bound, bound, Nx)
y = np.linspace(-bound, bound, Ny)
z = np.linspace(-bound, bound, Nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
dx = x[1] - x[0]
dy = y[1] - y[0]
dz = z[1] - z[0]

t0 = 0.0
psi = wavefunction(X, Y, Z, t0)
psi_magnitude = np.abs(psi)

positions = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T.astype(np.float32)
colors = compute_colors(psi)

# --- Global simulation parameters ---
dt = 1       # time step
angle = 0.0  # for simple rotation view
last_time = time.time()

# Global variables for rotation and zoom
rotation_x = 0.0
rotation_y = 0.0
last_mouse_x = 0
last_mouse_y = 0
mouse_down = False
zoom = 0.0  # zoom factor: positive values zoom in, negative zoom out

# --- Rotation Functions ---
def mouse_button(button, state, x, y):
    global mouse_down, last_mouse_x, last_mouse_y
    if button == GLUT_LEFT_BUTTON:
        if state == GLUT_DOWN:
            mouse_down = True
            last_mouse_x = x
            last_mouse_y = y
        elif state == GLUT_UP:
            mouse_down = False

def mouse_motion(x, y):
    global rotation_x, rotation_y, last_mouse_x, last_mouse_y
    if mouse_down:
        dx = x - last_mouse_x
        dy = y - last_mouse_y
        rotation_x += dy * 0.5
        rotation_y += dx * 0.5
        last_mouse_x = x
        last_mouse_y = y
        glutPostRedisplay()

# --- Keyboard Callback for Zoom ---
def keyboard(key, x, y):
    global zoom
    # GLUT passes key as bytes in Python 3, so decode it
    key = key.decode("utf-8") if isinstance(key, bytes) else key
    if key == '+' or key == '=':
        zoom += 1.0  # Zoom in
        glutPostRedisplay()
    elif key == '-' or key == '_':
        zoom -= 1.0  # Zoom out
        glutPostRedisplay()
    elif key == 'q' or key == 'Q':
        sys.exit(0)

# --- OpenGL / GLUT callbacks ---
def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    # Apply zoom by modifying the translation along the Z axis
    glTranslatef(0.0, 0.0, -20.0 + zoom)
    glRotatef(rotation_x, 1.0, 0.0, 0.0)
    glRotatef(rotation_y, 0.0, 1.0, 0.0)
    
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, positions)
    glColorPointer(4, GL_FLOAT, 0, colors)
    glDrawArrays(GL_POINTS, 0, positions.shape[0])
    glDisableClientState(GL_VERTEX_ARRAY)
    glDisableClientState(GL_COLOR_ARRAY)
    
    glutSwapBuffers()

def update(value):
    global psi, psi_magnitude, colors, last_time, angle, X, Y, Z, resolution
    
    current_time = time.time()
    dt_frame = current_time - last_time
    fps = 1.0 / dt_frame if dt_frame > 0 else 0.0
    last_time = current_time
    
    # Compute adaptive grid resolution based on wavefunction's magnitude
    X, Y, Z, resolution = create_adaptive_grid(psi_magnitude)

    # Compute the time evolution of the wavefunction
    psi = crank_nicolson_step(psi, dt)
    
    # Normalize wavefunction
    norm_val = np.sqrt(np.sum(np.abs(psi)**2))
    if norm_val != 0:
        psi = psi / norm_val
        
    psi_magnitude = np.abs(psi)
    
    # Compute colors based on the updated wavefunction
    colors = compute_colors(psi_magnitude)
    
    angle += 0.5
    if angle > 360:
        angle -= 360
    
    new_title = f"3D Wavefunction Simulation - FPS: {fps:.2f}"
    glutSetWindowTitle(new_title)
    
    glutPostRedisplay()
    glutTimerFunc(1, update, 0)

def reshape(width, height):
    if height == 0:
        height = 1
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, width / float(height), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)

def init_gl():
    glClearColor(0.1, 0.1, 0.1, 1.0)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_POINT_SMOOTH)
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
    glPointSize(3)

def main():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)
    glutInitWindowSize(800, 600)
    glutCreateWindow("3D Wavefunction Simulation - Zoom Enabled")
    
    init_gl()
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutMouseFunc(mouse_button)
    glutMotionFunc(mouse_motion)
    glutKeyboardFunc(keyboard)
    glutTimerFunc(1, update, 0)
    glutMainLoop()

if __name__ == '__main__':
    main()