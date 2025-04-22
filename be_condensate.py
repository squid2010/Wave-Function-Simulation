import glfw
from OpenGL.GL import *
import numpy as np
import ctypes
from pyrr import Matrix44
import imgui
from imgui.integrations.glfw import GlfwRenderer

# ---------------------------
# Simulation Parameters
# ---------------------------
Nx, Ny = 256, 256           # physics grid resolution
Lx, Ly = 20e-6, 20e-6       # domain size (m)
dx, dy = Lx/Nx, Ly/Ny

dt = 1e-6                  # time step (s)
adaptive_dt = True
slow_factor = 1             # evolution steps/frame
zoom = 1.0                  # 1.0 = full domain

# Constants (87Rb)
hbar = 1.0545718e-34
m = 1.44316060e-25
a_s = 5.3e-9
g_base = 4 * np.pi * hbar**2 * a_s / m
g = g_base

use_dissipation = False
gamma = 0.05
use_adaptive_dt = False
time_elapsed = 0.0
vortex_mode = False

# Trap & TF chemical potential
N_atoms = 1e5
omega_x = 2*np.pi*50
omega_y = 2*np.pi*50
omega_bar = np.sqrt(omega_x*omega_y)
a_ho = np.sqrt(hbar/(m*omega_bar))
mu = (hbar*omega_bar/2) * (15*N_atoms*a_s/a_ho)**(2/5)

# Classical turning radii
R_x = np.sqrt(2*mu/(m*omega_x**2))
R_y = np.sqrt(2*mu/(m*omega_y**2))

# Build simulation grid & potential
x = np.linspace(-Lx/2, Lx/2, Nx, dtype=np.float32)
y = np.linspace(-Ly/2, Ly/2, Ny, dtype=np.float32)
X, Y = np.meshgrid(x, y, indexing='ij')
V_ext = 0.5 * m * (omega_x**2 * X**2 + omega_y**2 * Y**2)

# Kinetic propagator factors
kx = 2*np.pi*np.fft.fftfreq(Nx, d=dx)
ky = 2*np.pi*np.fft.fftfreq(Ny, d=dy)
KX, KY = np.meshgrid(kx, ky, indexing='ij')
T = hbar*(KX**2 + KY**2)/(2*m)
kin_factor = np.exp(-1j * T * dt / slow_factor / hbar)

# Render modes
modes = ["Static", "Density", "Phase"]
mode_index = 1

def add_vortex(psi, X, Y, x0=0.0, y0=0.0, charge=1):
    angle = np.angle((X - x0) + 1j*(Y - y0))
    return psi * np.exp(1j * charge * angle)

def init_condensate():
    n_TF = np.maximum((mu - V_ext)/g_base, 0.0)
    psi = np.sqrt(n_TF).astype(np.complex128)
    norm = np.sqrt(np.sum(np.abs(psi)**2)*dx*dy)
    psi *= np.sqrt(N_atoms)/norm
    psi *= np.exp(1j*0.1*np.random.randn(*psi.shape))
    psi = add_vortex(psi, 0.0, 0.0, charge=1)
    return psi
    
def compute_total_energy(psi):
    psi_k = np.fft.fft2(psi)
    kinetic = np.sum((hbar**2/(2*m)) * np.abs(psi_k)**2 * (KX**2 + KY**2)) * dx * dy / (Nx * Ny)
    potential = np.sum(V_ext * np.abs(psi)**2) * dx * dy
    interaction = 0.5 * g * np.sum(np.abs(psi)**4) * dx * dy
    return kinetic, potential, interaction, kinetic + potential + interaction


def evolve(psi, dt_local, gval, gamma=0.0):
    psi_k = np.fft.fft2(psi)
    psi_k *= np.exp(-1j * T * dt_local / (2*hbar))
    psi = np.fft.ifft2(psi_k)

    abs2 = np.abs(psi)**2
    nonlinear = (V_ext + gval * abs2)
    
    # Split the evolution: real (dissipative) and imaginary (unitary) parts
    H = V_ext + gval * abs2
    psi *= np.exp(-gamma * H * dt_local / hbar)      # dissipative (norm reducing)
    psi *= np.exp(-1j * H * dt_local / hbar)         # unitary

    psi_k = np.fft.fft2(psi)
    psi_k *= np.exp(-1j * T * dt_local / (2*hbar))
    
    return np.fft.ifft2(psi_k)

# --------------------------------------
# Quad shaders (with trap‐mask)
# --------------------------------------
vertex_src = '''
#version 330 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aUV;
out vec2 vUV;
void main(){
    vUV = aUV;
    gl_Position = vec4(aPos,0,1);
}'''

fragment_src = '''
#version 330 core
in vec2 vUV;
out vec4 FragColor;

uniform sampler2D tex;
uniform int mode;
uniform float zoom;
uniform float Lx, Ly;
uniform float Rx, Ry;

vec3 hsv2rgb(vec3 c){
    vec3 p = abs(fract(c.xxx + vec3(0.0,2.0/3.0,1.0/3.0)) *6.0 -3.0) -1.0;
    return c.z * mix(vec3(1.0), clamp(p,0.0,1.0), c.y);
}

void main(){
    vec2 z = (vUV - 0.5)/zoom + 0.5;
    float X = (z.x - 0.5)*Lx;
    float Y = (z.y - 0.5)*Ly;
    if( (X*X)/(Rx*Rx) + (Y*Y)/(Ry*Ry) > 1.0 ){
        discard;
    }
    float v = texture(tex, z).r;
    if(mode == 1){
        FragColor = vec4(v,v,v,1);
    }
    else if(mode == 2){
        vec3 col = hsv2rgb(vec3(v,1,1));
        FragColor = vec4(col,1);
    }
    else {
        FragColor = vec4(v,v,v,1);
    }
}'''

def compile_shader(src, typ):
    sh = glCreateShader(typ)
    glShaderSource(sh, src)
    glCompileShader(sh)
    if not glGetShaderiv(sh, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(sh).decode())
    return sh

def main():
    global mode_index, dt, slow_factor, zoom, use_dissipation, use_adaptive_dt, gamma, time_elapsed

    if not glfw.init(): raise RuntimeError("GLFW init failed")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR,3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR,3)
    glfw.window_hint(glfw.OPENGL_PROFILE,glfw.OPENGL_CORE_PROFILE)
    win = glfw.create_window(1200,800,"BEC Simulator",None,None)
    glfw.make_context_current(win)
    glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)

    imgui.create_context(); impl = GlfwRenderer(win)

    vs = compile_shader(vertex_src, GL_VERTEX_SHADER)
    fs = compile_shader(fragment_src, GL_FRAGMENT_SHADER)
    prog = glCreateProgram()
    glAttachShader(prog, vs); glAttachShader(prog, fs)
    glLinkProgram(prog)
    loc_mode = glGetUniformLocation(prog, "mode")
    loc_zoom = glGetUniformLocation(prog, "zoom")
    loc_Lx   = glGetUniformLocation(prog, "Lx")
    loc_Ly   = glGetUniformLocation(prog, "Ly")
    loc_Rx   = glGetUniformLocation(prog, "Rx")
    loc_Ry   = glGetUniformLocation(prog, "Ry")

    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glBindTexture(GL_TEXTURE_2D, 0)

    quad_verts = np.array([
      -1,-1, 0,0,
       1,-1, 1,0,
       1, 1, 1,1,
      -1, 1, 0,1,
    ], dtype=np.float32)
    quad_idx = np.array([0,1,2, 2,3,0], dtype=np.uint32)
    quadVAO = glGenVertexArrays(1)
    quadVBO = glGenBuffers(1)
    quadEBO = glGenBuffers(1)
    glBindVertexArray(quadVAO)
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO)
    glBufferData(GL_ARRAY_BUFFER, quad_verts.nbytes, quad_verts, GL_STATIC_DRAW)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, quadEBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, quad_idx.nbytes, quad_idx, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,16,ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,16,ctypes.c_void_p(8))
    glBindVertexArray(0)

    psi0 = init_condensate()
    psi  = psi0.copy()
    
    click_pos = []
    
    def mouse_button_callback(window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
            xpos, ypos = glfw.get_cursor_pos(window)
            width, height = glfw.get_window_size(window)
    
            # Normalize to [0, 1]
            nx = xpos / width
            ny = ypos / height
            click_pos.append((nx, ny))
    glfw.set_mouse_button_callback(win, mouse_button_callback)

    def scroll_cb(w,x,y):
        global zoom
        zoom *= (1.1**y)
        zoom = np.clip(zoom, 0.1, 20.0)
    glfw.set_scroll_callback(win, scroll_cb)

    while not glfw.window_should_close(win):
        glfw.poll_events(); impl.process_inputs(); imgui.new_frame()

        imgui.begin("Controls")
        _, mode_index = imgui.combo("Mode", mode_index, modes)
        _, dt = imgui.input_float("dt (s)", dt, format="%.1e")
        _, slow_factor = imgui.slider_float("Slow Factor", slow_factor, 1, 1e10, format="%.0f")
        _, zoom = imgui.slider_float("Zoom", zoom, 0.1, 20.0, format="%.2f")
        _, use_dissipation = imgui.checkbox("Dissipation", use_dissipation)
        _, gamma = imgui.slider_float("Gamma", gamma, 0.0, 1.0)
        _, use_adaptive_dt = imgui.checkbox("Adaptive Timestep", use_adaptive_dt)
        if imgui.button("Add Vortex at Center"):
            psi = add_vortex(psi, X, Y)
        if imgui.button("Reset"):
            psi0 = init_condensate(); psi = psi0.copy()
        imgui.end()
        
        kin, pot, inter, total = compute_total_energy(psi)
        imgui.begin("Energy")
        imgui.text(f"Kinetic:     {kin:.2e}")
        imgui.text(f"Potential:   {pot:.2e}")
        imgui.text(f"Interaction: {inter:.2e}")
        imgui.text(f"Total:       {total:.2e}")
        imgui.end()

        if use_adaptive_dt:
            density = np.abs(psi)**2
            grad = np.max(np.abs(np.gradient(density)))
            dt_eff *= np.clip(1.0 / (1e3 * grad + 1e-6), 0.1, 10.0)

        kin_factor[:] = np.exp(-1j * T * dt / slow_factor / hbar)
        time_elapsed += dt
        gval = g * (1.0 + 0.2*np.sin(2*np.pi*0.5*time_elapsed))  # Example modulation
        dt_eff = dt / slow_factor
    
        if use_adaptive_dt:
            density = np.abs(psi)**2
            grad = np.max(np.abs(np.gradient(density)))
            dt_eff *= np.clip(1.0 / (1e3 * grad + 1e-6), 0.1, 10.0)
            
        if click_pos:
            width, height = glfw.get_window_size(win)
            aspect = width / height
            sim_width  = Lx / zoom
            sim_height = Ly / zoom
        
            for nx, ny in click_pos:
                py = (nx - 0.5) * sim_width
                px = (0.5 - ny) * sim_height
                psi = add_vortex(psi, X, Y, x0=px, y0=py, charge=1)
            click_pos.clear()
    
        if mode_index > 0:
            psi = evolve(psi, dt_eff, gval, gamma if use_dissipation else 0.0)
            if mode_index==1:
                vals = np.abs(psi)**2
            else:
                vals = (np.angle(psi)/(2*np.pi)+0.5)%1.0
        else:
            vals = np.abs(psi0)**2
        vals /= vals.max()
        vals_f = vals.astype(np.float32)

        glBindTexture(GL_TEXTURE_2D, tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, Nx, Ny, 0, GL_RED, GL_FLOAT, vals_f)
        glBindTexture(GL_TEXTURE_2D, 0)

        glClearColor(0.1,0.1,0.1,1); glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(prog)
        glUniform1i(loc_mode, mode_index)
        glUniform1f(loc_zoom, zoom)
        glUniform1f(loc_Lx, Lx)
        glUniform1f(loc_Ly, Ly)
        glUniform1f(loc_Rx, R_x)
        glUniform1f(loc_Ry, R_y)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex)
        glBindVertexArray(quadVAO)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

        imgui.render(); impl.render(imgui.get_draw_data()); glfw.swap_buffers(win)

    impl.shutdown(); glfw.terminate()

if __name__=="__main__":
    main()