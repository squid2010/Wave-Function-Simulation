import glfw
from OpenGL.GL import *
import numpy as np
import ctypes
from pyrr import Matrix44
import imgui
from imgui.integrations.glfw import GlfwRenderer
import metalcompute as mc  # use this if you wish to initialize on the GPU; falls back to CPU on error
from math import pi
import time

# ---------------------------
# Global Simulation Parameters (Macro-Level)
# ---------------------------
# Grid resolution & domain parameters (Cartesian grid)
Nx = 256         # Number of grid points along x
Ny = 256         # Number of grid points along y
Lx = 10.0        # Physical width of the domain in x
Ly = 10.0        # Physical height of the domain in y
dx = Lx / Nx
dy = Ly / Ny

# Time evolution parameters
dt = 0.001       # Time step
total_steps = 10000

# Physical constants for the GPE
hbar = 1.0
mass = 1.0
g = 1.0         # Interaction (nonlinearity) strength

# Uniform external potential (can be overridden by custom potential)
V_ext = np.zeros((Nx, Ny), dtype=np.float64)

# Evolution mode options
evolution_options = ["static", "phase", "full"]
current_evolution_index = 2  # default "full" evolution

# Control for simulation speed:
slow_factor = 1.0

# ---------------------------
# Initial Wavefunction Parameters (adjustable via GUI)
# ---------------------------
wave_x0 = 0.0      # initial center x-coordinate
wave_y0 = 0.0      # initial center y-coordinate
wave_sigma = 1.0   # initial width of Gaussian
wave_p0x = 5.0     # initial momentum in x
wave_p0y = 0.0     # initial momentum in y

# New parameter: Angular momentum (adds an extra phase L_ang * theta)
ang_mom = 0.0

# Damping parameter (γ). If > 0, each timestep multiplies ψ by exp(-γ dt)
damping = 0.0

# Custom potential: A text expression (in terms of x and y) to be added to V_ext.
# For example, "0.5 * (x**2+y**2)" for a harmonic trap.
use_custom_potential = False
V_custom_expr = "0.5 * (x**2+y**2)"  # default: no custom additional potential

# ---------------------------
# Optional: Parameters for Metal GPU Initialization
# ---------------------------
metal_shader = r'''
    #include <metal_stdlib>
    using namespace metal;
    
    // This kernel computes a 2D Gaussian on a Cartesian grid.
    kernel void computeGaussian2D(const device float* gridParams [[ buffer(0) ]],
                                  device float2* outWF [[ buffer(1) ]],
                                  constant float* params [[ buffer(2) ]],
                                  uint id [[ thread_position_in_grid ]]) {
        uint index = id * 2;
        float x = gridParams[index];
        float y = gridParams[index+1];
        float x0 = params[0];
        float y0 = params[1];
        float sigma = params[2];
        float p0x = params[3];
        float p0y = params[4];
        float L_ang = params[5];
        float A = 1.0 / (sqrt(2.0 * 3.141592653589793) * sigma);
        float dx = x - x0;
        float dy = y - y0;
        float r2 = dx*dx + dy*dy;
        float amplitude = A * exp(-r2/(2.0*sigma*sigma));
        float phase = p0x*dx + p0y*dy;
        // Add angular momentum phase:
        phase += L_ang * atan2(dy, dx);
        float psi_real = amplitude * cos(phase);
        float psi_imag = amplitude * sin(phase);
        outWF[id] = float2(psi_real, psi_imag);
    }
'''

# ---------------------------
# OpenGL Shader Helpers
# ---------------------------
def create_shader(shader_type, source):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(shader).decode())
    return shader

def create_program(vertex_source, fragment_source):
    program = glCreateProgram()
    vertex_shader = create_shader(GL_VERTEX_SHADER, vertex_source)
    fragment_shader = create_shader(GL_FRAGMENT_SHADER, fragment_source)
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)
    if not glGetProgramiv(program, GL_LINK_STATUS):
        raise RuntimeError(glGetProgramInfoLog(program).decode())
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)
    return program

# ---------------------------
# Shader Programs for Rendering
# ---------------------------
vertex_shader_source = """
    #version 330 core
    layout (location = 0) in vec2 position;
    layout (location = 1) in vec2 wf;
    out vec2 WaveFunction;
    uniform mat4 mvp;
    void main() {
        gl_Position = mvp * vec4(position, 0.0, 1.0);
        WaveFunction = wf;
    }
"""

fragment_shader_source = """
    #version 330 core
    in vec2 WaveFunction;
    out vec4 FragColor;
    uniform float maxIntensity;
    void main() {
        float intensity = dot(WaveFunction, WaveFunction);
        float phase = atan(WaveFunction.y, WaveFunction.x);
        float mappedIntensity = intensity / maxIntensity;
        float alpha = 1.0;
        if (intensity < maxIntensity / 4.0)
            alpha = mappedIntensity / 2.0;
        vec3 color = vec3(phase/6.2831 + 0.5, mappedIntensity, 1.0 - mappedIntensity);
        FragColor = vec4(color, alpha);
    }
"""

# ---------------------------
# Initialization of Wavefunction
# ---------------------------
def init_wavefunction_cpu():
    """
    Initialize the 2D wavefunction using a Gaussian whose phase includes both
    linear momentum and angular momentum.
    """
    x = np.linspace(-Lx/2, Lx/2, Nx, dtype=np.float32)
    y = np.linspace(-Ly/2, Ly/2, Ny, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing='ij')
    global wave_x0, wave_y0, wave_sigma, wave_p0x, wave_p0y, ang_mom
    x0, y0 = wave_x0, wave_y0
    sigma = wave_sigma
    p0x, p0y = wave_p0x, wave_p0y
    A = 1.0 / (np.sqrt(2*np.pi) * sigma)
    amplitude = A * np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
    # Linear momentum phase plus angular momentum phase:
    phase = p0x * (X - x0) + p0y * (Y - y0) + ang_mom * np.arctan2(Y - y0, X - x0)
    psi = amplitude * np.exp(1j * phase)
    norm = np.sqrt(np.sum(np.abs(psi)**2) * dx * dy)
    psi /= norm
    return psi

def init_wavefunction_metal():
    """
    Try using the Metal compute shader for initialization.
    Falls back to CPU if any error occurs.
    """
    try:
        global wave_x0, wave_y0, wave_sigma, wave_p0x, wave_p0y, ang_mom
        xs = np.linspace(-Lx/2, Lx/2, Nx, dtype=np.float32)
        ys = np.linspace(-Ly/2, Ly/2, Ny, dtype=np.float32)
        X, Y = np.meshgrid(xs, ys, indexing='ij')
        grid = np.column_stack((X.ravel(), Y.ravel()))
        num_points = grid.shape[0]
        in_data = grid.astype(np.float32).ravel().tobytes()
        dev = mc.Device()
        in_buf = dev.buffer(in_data)
        out_size = num_points * 2 * 4
        out_buf = dev.buffer(out_size)
        params = np.array([wave_x0, wave_y0, wave_sigma, wave_p0x, wave_p0y, ang_mom],
                          dtype=np.float32).tobytes()
        param_buf = dev.buffer(params)
        kernel_fn = dev.kernel(metal_shader).function("computeGaussian2D")
        kernel_fn(num_points, in_buf, out_buf, param_buf)
        computed_data = np.array(memoryview(out_buf).cast('f'))
        wf = computed_data.reshape((num_points, 2))
        psi = wf[:, 0] + 1j * wf[:, 1]
        psi = psi.reshape((Nx, Ny))
        norm = np.sqrt(np.sum(np.abs(psi)**2) * dx * dy)
        psi /= norm
        return psi
    except Exception as e:
        print("Metal initialization failed, falling back to CPU:", e)
        return init_wavefunction_cpu()

# Global arrays for grid coordinates (used for custom potential evaluation)
Xgrid = None
Ygrid = None

# ---------------------------
# Precompute Kinetic Propagation Factor (Fourier space)
# ---------------------------
def compute_kinetic_factor():
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    k2 = KX**2 + KY**2
    T = (hbar**2 * k2) / (2 * mass)
    return np.exp(-1j * T * dt / (2 * hbar))

# ---------------------------
# Nonlinear Propagation Factor
# ---------------------------
def nonlinear_factor(psi):
    """
    Compute exp(-i*(V_ext + g|ψ|² + V_custom)*dt/hbar).
    If a custom potential expression is enabled, evaluate it on the grid.
    """
    V_eff = V_ext + g * np.abs(psi)**2
    if use_custom_potential and V_custom_expr.strip() != "":
        # Safe evaluation dictionary: allow np and provide grid arrays.
        safe_dict = {"np": np}
        # Ensure Xgrid and Ygrid are defined.
        global Xgrid, Ygrid
        if Xgrid is not None and Ygrid is not None:
            try:
                # Evaluate the custom potential expression.
                V_custom = eval(V_custom_expr, safe_dict, {"x": Xgrid, "y": Ygrid})
                V_eff = V_eff + V_custom
            except Exception as e:
                print("Error evaluating custom potential:", e)
    return np.exp(-1j * V_eff * dt / hbar)

# ---------------------------
# Time Evolution using Split-Step Fourier Method
# ---------------------------
def evolve_wavefunction(psi, kinetic_factor):
    psi_k = np.fft.fft2(psi)
    psi_k *= kinetic_factor
    psi = np.fft.ifft2(psi_k)
    psi *= nonlinear_factor(psi)
    psi_k = np.fft.fft2(psi)
    psi_k *= kinetic_factor
    psi = np.fft.ifft2(psi_k)
    # Apply damping if nonzero.
    if damping > 0:
        psi *= np.exp(-damping * dt)
    norm = np.sqrt(np.sum(np.abs(psi)**2) * dx * dy)
    psi /= norm
    return psi

# ---------------------------
# OpenGL Buffer Setup for Rendering
# ---------------------------
def create_grid_vbo():
    global Xgrid, Ygrid
    xs = np.linspace(-Lx/2, Lx/2, Nx, dtype=np.float32)
    ys = np.linspace(-Ly/2, Ly/2, Ny, dtype=np.float32)
    Xgrid, Ygrid = np.meshgrid(xs, ys, indexing='ij')
    positions = np.column_stack((Xgrid.ravel(), Ygrid.ravel())).astype(np.float32)
    wf_data = np.zeros((positions.shape[0], 2), dtype=np.float32)
    interleaved = np.empty(positions.shape[0], dtype=[('pos', np.float32, 2), ('wf', np.float32, 2)])
    interleaved['pos'] = positions
    interleaved['wf'] = wf_data
    return interleaved

# ---------------------------
# Main Program
# ---------------------------
def main():
    global evolution_options, current_evolution_index, slow_factor, V_ext, g, dt
    global wave_x0, wave_y0, wave_sigma, wave_p0x, wave_p0y, ang_mom, damping, use_custom_potential, V_custom_expr
    window_width, window_height = 1200, 800
    if not glfw.init():
        raise Exception("GLFW initialization failed")
    
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    
    window = glfw.create_window(window_width, window_height, "Macro-Scale Quantum Fluid Simulation", None, None)
    if not window:
        glfw.terminate()
        raise Exception("GLFW window creation failed")
    
    glfw.make_context_current(window)
    glDisable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glClearColor(0.225, 0.215, 0.2, 1.0)
    glPointSize(3)
    
    imgui.create_context()
    impl = GlfwRenderer(window)
    
    # placeholder for ImGui’s old callback
    prev_key_cb = None

    # 1) define your chained key callback first
    def key_callback(window, key, scancode, action, mods):
        # let ImGui handle every key first
        if prev_key_cb is not None:
            prev_key_cb(window, key, scancode, action, mods)

        # then do your dt tweaking
        global dt
        if action in (glfw.PRESS, glfw.REPEAT):
            if key == glfw.KEY_UP:
                dt *= 1.1
            elif key == glfw.KEY_DOWN:
                dt /= 1.1

    # 2) install it, capturing whatever was there before
    prev_key_cb = glfw.set_key_callback(window, key_callback)
    
    # Use an orthographic projection (with identity view) to map [-Lx/2,Lx/2] and [-Ly/2,Ly/2] directly.
    projection = Matrix44.orthogonal_projection(-Lx/2, Lx/2, -Ly/2, Ly/2, -1.0, 1.0)
    mvp = projection
    
    shader_program = create_program(vertex_shader_source, fragment_shader_source)
    glUseProgram(shader_program)
    
    psi = init_wavefunction_metal()  # try GPU; falls back to CPU
    kinetic_factor = compute_kinetic_factor()
    
    interleaved = create_grid_vbo()
    psi_flat = np.column_stack((np.real(psi).ravel(), np.imag(psi).ravel())).astype(np.float32)
    interleaved['wf'] = psi_flat
    num_points = interleaved.shape[0]
    
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, interleaved.nbytes, interleaved, GL_DYNAMIC_DRAW)
    
    pos_loc = 0
    wf_loc = 1
    stride = interleaved.dtype.itemsize
    glVertexAttribPointer(pos_loc, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(pos_loc)
    offset = 2 * ctypes.sizeof(ctypes.c_float)
    glVertexAttribPointer(wf_loc, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset))
    glEnableVertexAttribArray(wf_loc)
    
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)
    
    maxIntensity = np.max(np.abs(psi)**2)
    glUseProgram(shader_program)
    glUniform1f(glGetUniformLocation(shader_program, "maxIntensity"), maxIntensity)
    
    
    last_frame_time = glfw.get_time()
    fps_last_time = last_frame_time
    frame_count = 0
    
    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()
        imgui.new_frame()
        
        win_w, win_h = glfw.get_window_size(window)
        panel_width = 300.0
        imgui.set_next_window_position(win_w - panel_width, 0)
        imgui.set_next_window_size(panel_width, win_h)
        imgui.begin("Controls", flags=imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE)
        
        changed, current_evolution_index = imgui.combo("Evolution Mode", current_evolution_index, evolution_options)
        evolution_mode = evolution_options[current_evolution_index]
        changed, slow_factor = imgui.input_float("Slow factor", slow_factor)
        
        changed, g_val = imgui.input_float("Interaction g", g)
        if changed:
            g = g_val
        changed, potential_val = imgui.input_float("Uniform V", V_ext[0,0])
        if changed:
            V_ext.fill(potential_val)
        changed, dt_val = imgui.input_float("Time step dt", dt)
        if changed:
            dt = dt_val
            kinetic_factor = compute_kinetic_factor()
        
        # Controls for initial wavefunction parameters.
        changed, new_wave_x0 = imgui.input_float("Initial x0", wave_x0)
        if changed: wave_x0 = new_wave_x0
        changed, new_wave_y0 = imgui.input_float("Initial y0", wave_y0)
        if changed: wave_y0 = new_wave_y0
        changed, new_wave_sigma = imgui.input_float("Sigma", wave_sigma)
        if changed: wave_sigma = new_wave_sigma
        changed, new_wave_p0x = imgui.input_float("Momentum p0x", wave_p0x)
        if changed: wave_p0x = new_wave_p0x
        changed, new_wave_p0y = imgui.input_float("Momentum p0y", wave_p0y)
        if changed: wave_p0y = new_wave_p0y
        
        # New control for angular momentum.
        changed, new_ang_mom = imgui.input_float("Angular momentum", ang_mom)
        if changed: ang_mom = new_ang_mom
        
        # New control for damping/dissipation.
        changed, new_damp = imgui.input_float("Damping", damping)
        if changed: damping = new_damp
        
        # New controls for custom potential.
        changed, use_custom = imgui.checkbox("Use custom potential", use_custom_potential)
        use_custom_potential = use_custom
        changed, new_Vcustom = imgui.input_text("V(x,y) expression", V_custom_expr, 256)
        if changed:
            V_custom_expr = new_Vcustom
        
        if imgui.button("Reset Wavefunction"):
            psi = init_wavefunction_metal()
            kinetic_factor = compute_kinetic_factor()
        
        imgui.end()
        
        current_time = glfw.get_time()
        elapsed = current_time - last_frame_time
        last_frame_time = current_time
        
        if evolution_mode == "static":
            pass
        elif evolution_mode == "phase":
            psi *= np.exp(-1j * dt)
        elif evolution_mode == "full":
            psi = evolve_wavefunction(psi, kinetic_factor**slow_factor)
        
        maxIntensity = np.max(np.abs(psi)**2)
        glUseProgram(shader_program)
        glUniform1f(glGetUniformLocation(shader_program, "maxIntensity"), maxIntensity)
        
        psi_flat = np.column_stack((np.real(psi).ravel(), np.imag(psi).ravel())).astype(np.float32)
        interleaved['wf'] = psi_flat
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, interleaved.nbytes, interleaved)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(shader_program)
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "mvp"), 1, GL_FALSE, mvp.astype(np.float32))
        glBindVertexArray(VAO)
        glDrawArrays(GL_POINTS, 0, num_points)
        glBindVertexArray(0)
        
        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)
        
        frame_count += 1
        if current_time - fps_last_time >= 1.0:
            fps = frame_count / (current_time - fps_last_time)
            glfw.set_window_title(window, f"Macro Quantum Fluid - FPS: {fps:.2f}")
            frame_count = 0
            fps_last_time = current_time

    impl.shutdown()
    glfw.terminate()

if __name__ == "__main__":
    main()