import sys
import time
import ctypes
import numpy as np
from numba import njit, prange
from scipy.special import hermite, factorial
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

# --- Simulation functions (unchanged) ---
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
    r = np.sqrt(x**2 + y**2 + z**2)+1e-10
    
    sin_theta = np.sqrt(x**2 + y**2) / r
    
    return 1/(8*np.sqrt(np.pi)*a0**(3/2))*np.exp(-r/2/a0)*sin_theta*np.exp(1j * np.arctan2(y, x))

@njit(parallel=True)
def potential(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    epsilon = 1e-4
    return -1.0 / np.sqrt(r**2 + epsilon)

def crank_nicolson_step(psi, tol=1e-6, max_iter=10):
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

def compute_colors(psi, alpha_exp=2.0):
    prob_density = np.abs(psi)**2
    max_val = np.max(prob_density)
    normed = prob_density / max_val if max_val > 0 else prob_density
    t = normed.ravel()
    lowColor = np.array([0.0, 0.0, 0.12, 0.0])
    highColor = np.array([1.0, 0.95, 0.88, 0.1])
    # Here we blend the colors linearly.
    rgb = (1 - t)[:, None] * lowColor[:3] + t[:, None] * highColor[:3]
    alpha = ((1 - t)[:, None] * lowColor[3] + t[:, None] * highColor[3]) * (t ** alpha_exp)[:, None]
    colors = np.hstack((rgb, alpha)).astype(np.float32)
    return colors

# --- Grid and initial conditions ---
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
positions = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T.astype(np.float32)
colors = compute_colors(psi)

# --- Global simulation parameters ---
dt = 1.0   # Time step
last_time = time.time()

# Global variables for rotation, zoom, and mouse state
rotation_x = 0.0
rotation_y = 0.0
last_mouse_x = 0
last_mouse_y = 0
mouse_down = False
zoom = 0.0

# --- Shader for point sprites ---
shader_program = None

def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        print("Shader compile error:", glGetShaderInfoLog(shader))
        return None
    return shader

def init_shader():
    global shader_program
    # Vertex shader: pass position and color, and set point size.
    vertex_src = """
    #version 120
    attribute vec3 position;
    attribute vec4 vertexColor;
    varying vec4 fragColor;
    void main(){
        fragColor = vertexColor;
        gl_Position = gl_ModelViewProjectionMatrix * vec4(position, 1.0);
        gl_PointSize = 8.0;
    }
    """
    # Fragment shader: use gl_PointCoord to produce a smooth circular point.
    fragment_src = """
    #version 120
    varying vec4 fragColor;
    void main(){
        vec2 coord = gl_PointCoord - vec2(0.5);
        float dist = length(coord);
        float alpha = smoothstep(0.5, 0.45, dist);
        gl_FragColor = vec4(fragColor.rgb, fragColor.a * alpha);
    }
    """
    vert_shader = compile_shader(vertex_src, GL_VERTEX_SHADER)
    frag_shader = compile_shader(fragment_src, GL_FRAGMENT_SHADER)
    shader_program = glCreateProgram()
    glAttachShader(shader_program, vert_shader)
    glAttachShader(shader_program, frag_shader)
    glBindAttribLocation(shader_program, 0, b'position')
    glBindAttribLocation(shader_program, 1, b'vertexColor')
    glLinkProgram(shader_program)
    if glGetProgramiv(shader_program, GL_LINK_STATUS) != GL_TRUE:
        print("Shader linking error:", glGetProgramInfoLog(shader_program))
    glDeleteShader(vert_shader)
    glDeleteShader(frag_shader)

# --- Input Callbacks ---
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
        dx_mouse = x - last_mouse_x
        dy_mouse = y - last_mouse_y
        rotation_x += dy_mouse * 0.5
        rotation_y += dx_mouse * 0.5
        last_mouse_x = x
        last_mouse_y = y
        glutPostRedisplay()

def keyboard(key, x, y):
    global zoom
    key = key.decode("utf-8") if isinstance(key, bytes) else key
    if key in ['+', '=']:
        zoom += 1.0
        glutPostRedisplay()
    elif key in ['-', '_']:
        zoom -= 1.0
        glutPostRedisplay()
    elif key in ['q', 'Q']:
        sys.exit(0)

# --- OpenGL / GLUT callbacks ---
def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    glTranslatef(0.0, 0.0, -20.0 + zoom)
    glRotatef(rotation_x, 1.0, 0.0, 0.0)
    glRotatef(rotation_y, 0.0, 1.0, 0.0)
    
    glUseProgram(shader_program)
    # Enable the vertex attributes.
    glEnableVertexAttribArray(0)  # position
    glEnableVertexAttribArray(1)  # vertexColor
    # Pass position data (using pointer to the numpy array data).
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, positions)
    # Pass color data.
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, colors)
    
    glDrawArrays(GL_POINTS, 0, positions.shape[0])
    
    glDisableVertexAttribArray(0)
    glDisableVertexAttribArray(1)
    glUseProgram(0)
    
    glutSwapBuffers()

def update(value):
    global psi, colors, last_time, rotation_x, rotation_y, zoom
    current_time = time.time()
    dt_frame = current_time - last_time
    last_time = current_time
    
    psi = crank_nicolson_step(psi)
    norm_val = np.sqrt(np.sum(np.abs(psi)**2))
    if norm_val != 0:
        psi = psi / norm_val
    colors = compute_colors(psi)
    
    glutSetWindowTitle(f"3D Wavefunction Simulation - FPS: {1.0/dt_frame:.2f}")
    glutPostRedisplay()
    glutTimerFunc(1, update, 0)

def reshape(width, height):
    if height == 0:
        height = 1
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, width/float(height), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)

def init_gl():
    glClearColor(0.1, 0.1, 0.1, 1.0)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_POINT_SMOOTH)
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
    glEnable(GL_PROGRAM_POINT_SIZE)
    glPointSize(6)  # Base point size; actual size set in shader.
    init_shader()

def main():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)
    glutInitWindowSize(800, 600)
    glutCreateWindow("3D Wavefunction Simulation - Continuous Cloud")
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
