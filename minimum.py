import glfw
from OpenGL.GL import *
import numpy as np
import ctypes
from pyrr import Matrix44, Quaternion
import metalcompute as mc
import imgui
from imgui.integrations.glfw import GlfwRenderer
from math import exp

# ---------------------------
# Global Simulation Parameters
# ---------------------------
num_r = 50         # radial grid points
num_theta = 100    # angular grid (theta)
num_phi = 100      # angular grid (phi)

R_box = 10.0
rcut_percent = 0.9  # R_cut = rcut_percent * R_box

wavefunction_options = ["gaussian"]
current_wavefunction_index = 0
wavefunction_type = wavefunction_options[current_wavefunction_index]

evolution_options = ["static", "phase", "full"]
current_evolution_index = 0  # default "static"
evolution_mode = evolution_options[current_evolution_index]

slow_factor = 1.0

# For Gaussian
x0_vals = [0.0, 0.0, 0.0]
sigma_val = 1.0
p0_vals = [10.0, 0.0, 0.0]


# Parameters for the potential (for CN update)
V0 = 1e6  # large potential outside the box
near_val = 0.01
far_val = 1000.0
R_cut = rcut_percent * R_box

# ---------------------------
# OpenGL Helper Functions
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
# Metal Compute Shader (for precomputing the initial wavefunction)
# ---------------------------
metal_shader = r'''
    #include <metal_stdlib>
    using namespace metal;
    
    kernel void computeGaussianWavepacket(const device float* inCoords [[ buffer(0) ]],
                                            device float2* outWF [[ buffer(1) ]],
                                            constant float* params [[ buffer(2) ]],
                                            uint id [[ thread_position_in_grid ]]) {
        uint index = id * 3;
        float r = inCoords[index];
        float theta = inCoords[index + 1];
        float phi = inCoords[index + 2];
        
        float x = r * sin(theta) * cos(phi);
        float y = r * sin(theta) * sin(phi);
        float z = r * cos(theta);
        float3 pos = float3(x, y, z);
        
        float3 center = float3(params[0], params[1], params[2]);
        float3 diff = pos - center;
        float r2 = dot(diff, diff);
        
        const float PI = 3.141592653589793;
        float sigma = params[3];
        float A = 1.0 / (pow(PI, 0.75) * pow(sigma, 1.5));
        float amplitude = A * exp(-r2 / (2.0 * sigma * sigma));
        
        float3 p0 = float3(params[4], params[5], params[6]);
        float phase = dot(p0, diff);
        
        float psi_real = amplitude * cos(phase);
        float psi_imag = amplitude * sin(phase);
        
        outWF[id] = float2(psi_real, psi_imag);
    }
'''

# ---------------------------
# Precompute Wavefunctions
# ---------------------------
def precompute_wavefunctions(spherical_coords):
    global x0_vals, sigma_val, p0_vals, wavefunction_type
    num_points = spherical_coords.shape[0]
    in_data = spherical_coords.astype(np.float32).ravel().tobytes()
    dev = mc.Device()
    in_buf = dev.buffer(in_data)
    out_size = num_points * 2 * 4  # 2 floats per point
    out_buf = dev.buffer(out_size)
    if wavefunction_type == "gaussian":
        params = np.array([x0_vals[0], x0_vals[1], x0_vals[2],
                           sigma_val, p0_vals[0], p0_vals[1], p0_vals[2]],
                          dtype=np.float32).tobytes()
        param_buf = dev.buffer(params)
        kernel_fn = dev.kernel(metal_shader).function("computeGaussianWavepacket")
    kernel_fn(num_points, in_buf, out_buf, param_buf)
    computed_data = np.array(memoryview(out_buf).cast('f'))
    precomputed_wf = computed_data.reshape(-1, 2)
    return precomputed_wf

def compute_max_intensity_cpu(precomputed_wf):
    intensities = np.sum(precomputed_wf**2, axis=1)
    return np.max(intensities)

# ---------------------------
# Shaders for Rendering
# ---------------------------
vertex_shader_source = """
    #version 330 core
    layout (location = 0) in vec3 position;
    layout (location = 1) in vec2 precomputedWF;
    out vec2 WaveFunction;
    uniform mat4 mvp;
    void main() {
        gl_Position = mvp * vec4(position, 1.0);
        WaveFunction = precomputedWF;
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
        float brightness = 1.0;
        if (intensity < maxIntensity / 4.0)
            alpha = mappedIntensity / 2.0;
        vec3 objectColor = vec3(phase / 10.0, mappedIntensity, mappedIntensity);
        FragColor = vec4(objectColor, alpha) * brightness;
    }
"""

# ---------------------------
# Main Program
# ---------------------------
def main():
    global wavefunction_type, evolution_mode, x0_vals, sigma_val, p0_vals
    global R_box, rcut_percent, slow_factor, R_cut
    global current_wavefunction_index, current_evolution_index

    window_width, window_height = 1200, 800
    if not glfw.init():
        raise Exception("GLFW initialization failed")
    
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    
    window = glfw.create_window(window_width, window_height, "Wavefunction Evolution", None, None)
    if not window:
        glfw.terminate()
        raise Exception("GLFW window creation failed")
    
    glfw.make_context_current(window)
    glDisable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glClearColor(0.225, 0.215, 0.2, 1.0)
    
    # Initialize Dear ImGui
    imgui.create_context()
    impl = GlfwRenderer(window)
    
    rotation = Matrix44.identity()
    lastPos = None
    mouse_pressed = False
    fov = 45.0  # field-of-view for zoom control

    def key_callback(window, key, scancode, action, mods):
        nonlocal fov
        if action in (glfw.PRESS, glfw.REPEAT):
            if key in (glfw.KEY_KP_ADD, glfw.KEY_EQUAL):
                fov -= 1.0
            elif key in (glfw.KEY_KP_SUBTRACT, glfw.KEY_MINUS):
                fov += 1.0
            fov = max(1.0, min(fov, 120.0))
    
    glfw.set_key_callback(window, key_callback)

    def trackball_mapping(x, y, width, height):
        v = np.array([2.0*x/width - 1.0, 1.0 - 2.0*y/height, 0.0], dtype=np.float32)
        d = np.linalg.norm(v[:2])
        if d < 1.0:
            v[2] = np.sqrt(1.0 - d*d)
        else:
            v[:2] /= d
        return v
    
    def mouse_button_callback(window, button, action, mods):
        nonlocal mouse_pressed, lastPos
        if button == glfw.MOUSE_BUTTON_LEFT:
            mouse_pressed = (action == glfw.PRESS)
            if mouse_pressed:
                width, height = glfw.get_framebuffer_size(window)
                x, y = glfw.get_cursor_pos(window)
                lastPos = trackball_mapping(x, y, width, height)
            else:
                lastPos = None
    
    def cursor_position_callback(window, xpos, ypos):
        nonlocal rotation, lastPos, mouse_pressed
        if not mouse_pressed:
            return
        width, height = glfw.get_framebuffer_size(window)
        currentPos = trackball_mapping(xpos, ypos, width, height)
        if lastPos is None:
            lastPos = currentPos
            return
        axis = np.cross(currentPos, lastPos)
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-6:
            axis /= axis_norm
            dot_val = np.clip(np.dot(lastPos, currentPos), -1.0, 1.0)
            angle = np.arccos(dot_val) * 2.0  # sensitivity factor
            delta_quat = Quaternion.from_axis_rotation(axis, angle)
            delta_mat = Matrix44.from_quaternion(delta_quat)
            rotation = delta_mat * rotation
        lastPos = currentPos
        
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_position_callback)

    # Compute spherical coordinates and Cartesian positions.
    r_vals = np.linspace(0, R_box, num_r)
    theta_vals = np.linspace(0, 2*np.pi, num_theta)
    phi_vals = np.linspace(0, np.pi, num_phi)
    r, theta, phi = np.meshgrid(r_vals, theta_vals, phi_vals, indexing='ij')
    spherical_coords = np.column_stack((r.ravel(), theta.ravel(), phi.ravel())).astype(np.float32)
    # Save r values for later sound computation.
    r_values = spherical_coords[:,0].copy()
    x = spherical_coords[:,0] * np.sin(spherical_coords[:,2]) * np.cos(spherical_coords[:,1])
    y = spherical_coords[:,0] * np.sin(spherical_coords[:,2]) * np.sin(spherical_coords[:,1])
    z = spherical_coords[:,0] * np.cos(spherical_coords[:,2])
    positions = np.column_stack((x, y, z)).astype(np.float32)

    precomputed_wf = precompute_wavefunctions(spherical_coords)
    maxIntensity = compute_max_intensity_cpu(precomputed_wf)
    num_points = positions.shape[0]
    interleaved = np.empty(num_points, dtype=[('pos', np.float32, 3),
                                                ('wf',  np.float32, 2)])
    interleaved['pos'] = positions
    interleaved['wf'] = precomputed_wf

    shader_program = create_program(vertex_shader_source, fragment_shader_source)
    glUseProgram(shader_program)
    glUniform1f(glGetUniformLocation(shader_program, "maxIntensity"), maxIntensity)
    
    psi_complex = precomputed_wf[:,0] + 1j * precomputed_wf[:,1]
    atom_time = 0.0
    
    view = Matrix44.look_at(eye=[R_box*1.5, R_box*1.5, R_box*1.5],
                            target=[0, 0, 0], up=[0, 1, 0])
    
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)
    interleavedVBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, interleavedVBO)
    glBufferData(GL_ARRAY_BUFFER, interleaved.nbytes, interleaved, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, interleaved.dtype.itemsize, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, interleaved.dtype.itemsize,
                          ctypes.c_void_p(3 * ctypes.sizeof(ctypes.c_float)))
    glEnableVertexAttribArray(1)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    glPointSize(10)
    last_frame_time = glfw.get_time()
    fps_last_time = last_frame_time
    frame_count = 0

    # Variables for sound update timing.
    last_sound_update = glfw.get_time()
    sound_update_interval = 0.1  # seconds

    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()
        imgui.new_frame()
        
        # Place GUI panel on the right.
        win_w, win_h = glfw.get_window_size(window)
        panel_width = 300.0
        imgui.set_next_window_position(win_w - panel_width, 0)
        imgui.set_next_window_size(panel_width, win_h)
        
        imgui.begin("Parameters", flags=imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE)
        changed, current_wavefunction_index = imgui.combo("Wavefunction Type", current_wavefunction_index, wavefunction_options)
        wavefunction_type = wavefunction_options[current_wavefunction_index]
        changed, current_evolution_index = imgui.combo("Evolution Mode", current_evolution_index, evolution_options)
        evolution_mode = evolution_options[current_evolution_index]
        
        changed, slow_factor = imgui.input_float("slow factor", slow_factor)
        changed, R_box = imgui.input_float("R_box", R_box)
        changed, rcut_percent = imgui.input_float("R_cut (%)", rcut_percent)
        
        if wavefunction_type == "gaussian":
            changed, x0_val = imgui.input_float("x0", x0_vals[0])
            if changed: x0_vals[0] = x0_val
            changed, y0_val = imgui.input_float("y0", x0_vals[1])
            if changed: x0_vals[1] = y0_val
            changed, z0_val = imgui.input_float("z0", x0_vals[2])
            if changed: x0_vals[2] = z0_val
            changed, sigma_val = imgui.input_float("sigma", sigma_val)
            changed, p0x = imgui.input_float("p0_x", p0_vals[0])
            if changed: p0_vals[0] = p0x
            changed, p0y = imgui.input_float("p0_y", p0_vals[1])
            if changed: p0_vals[1] = p0y
            changed, p0z = imgui.input_float("p0_z", p0_vals[2])
            if changed: p0_vals[2] = p0z


        if imgui.button("Update Wavefunction"):
            R_cut = rcut_percent * R_box
            r_vals = np.linspace(0, R_box, num_r)
            theta_vals = np.linspace(0, 2*np.pi, num_theta)
            phi_vals = np.linspace(0, np.pi, num_phi)
            r, theta, phi = np.meshgrid(r_vals, theta_vals, phi_vals, indexing='ij')
            spherical_coords = np.column_stack((r.ravel(), theta.ravel(), phi.ravel())).astype(np.float32)
            # Update r_values as well
            r_values = spherical_coords[:,0].copy()
            x = spherical_coords[:,0] * np.sin(spherical_coords[:,2]) * np.cos(spherical_coords[:,1])
            y = spherical_coords[:,0] * np.sin(spherical_coords[:,2]) * np.sin(spherical_coords[:,1])
            z = spherical_coords[:,0] * np.cos(spherical_coords[:,2])
            positions = np.column_stack((x, y, z)).astype(np.float32)
            interleaved['pos'] = positions
            
            precomputed_wf = precompute_wavefunctions(spherical_coords)
            psi_complex = precomputed_wf[:,0] + 1j * precomputed_wf[:,1]
            atom_time = 0.0
            maxIntensity = compute_max_intensity_cpu(precomputed_wf)
            glUseProgram(shader_program)
            glUniform1f(glGetUniformLocation(shader_program, "maxIntensity"), maxIntensity)
            interleaved['wf'] = precomputed_wf
            glBindBuffer(GL_ARRAY_BUFFER, interleavedVBO)
            glBufferSubData(GL_ARRAY_BUFFER, 0, interleaved.nbytes, interleaved)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
        imgui.end()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        projection = Matrix44.perspective_projection(fov, window_width/float(window_height), near_val, far_val)
        model = rotation
        mvp = projection * view * model
        glUseProgram(shader_program)
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "mvp"), 1, GL_FALSE, mvp.astype(np.float32))
        
        current_time = glfw.get_time()
        dt = current_time - last_frame_time
        last_frame_time = current_time
        
        # Evolution update:
        if evolution_mode == "static":
            psi_flat = psi_complex.ravel()
        elif evolution_mode == "phase":
            if wavefunction_type == "gaussian":
                E = 1.0
                lam = (1 - 1j*dt*E/2) / (1 + 1j*dt*E/2)
                psi_complex *= lam
            psi_flat = psi_complex.ravel()
        elif evolution_mode == "full":
            # For full evolution we update spatially using finite differences.
            if wavefunction_type == "gaussian":
                dt_eff = dt * slow_factor
                psi_complex = psi_complex.reshape((num_r, num_theta*num_phi))
                dr = r_vals[1] - r_vals[0]
                N = num_r - 1
                r_interior = r_vals[:N]
                V = np.where(r_interior < R_cut, 0.0, V0)
                a = np.zeros(N-1, dtype=float)
                b = np.zeros(N, dtype=float)
                c = np.zeros(N-1, dtype=float)
                b[0] = -1.0/(dr**2) + V[0]
                c[0] = 1.0/(dr**2)
                for i in range(1, N):
                    a[i-1] = -1.0/(2*dr**2) + 1.0/(4*r_interior[i]*dr)
                    b[i] = 1.0/(dr**2) + V[i]
                    if i < N:
                        c[i-1] = -1.0/(2*dr**2) - 1.0/(4*r_interior[i]*dr)
                A_diag = 1 + 1j*dt_eff/2 * b
                B_diag = 1 - 1j*dt_eff/2 * b
                A = np.diag(A_diag) + np.diag(1j*dt_eff/2 * a, k=-1) + np.diag(1j*dt_eff/2 * c, k=1)
                B = np.diag(B_diag) + np.diag(-1j*dt_eff/2 * a, k=-1) + np.diag(-1j*dt_eff/2 * c, k=1)
                psi_old = psi_complex[:N, :].copy()
                d_rhs = B.dot(psi_old)
                psi_new = np.linalg.solve(A, d_rhs)
                psi_updated = np.zeros_like(psi_complex, dtype=complex)
                psi_updated[:N, :] = psi_new
                psi_updated[N, :] = 0.0
                psi_complex = psi_updated
                psi_flat = psi_complex.ravel()
            else:
                psi_flat = psi_complex.ravel()
        else:
            psi_flat = psi_complex.ravel()
        
        wf_data = np.column_stack((np.real(psi_flat), np.imag(psi_flat))).astype(np.float32)
        interleaved['wf'] = wf_data
        glBindBuffer(GL_ARRAY_BUFFER, interleavedVBO)
        glBufferSubData(GL_ARRAY_BUFFER, 0, interleaved.nbytes, interleaved)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        
        glBindVertexArray(VAO)
        glDrawArrays(GL_POINTS, 0, num_points)
        glBindVertexArray(0)

        imgui.render()
        impl.render(imgui.get_draw_data())
        
        glfw.swap_buffers(window)
        frame_count += 1
        if current_time - fps_last_time >= 1.0:
            fps = frame_count / (current_time - fps_last_time)
            glfw.set_window_title(window, f"Wavefunction - FPS: {fps:.2f}")
            frame_count = 0
            fps_last_time = current_time

    impl.shutdown()
    glfw.terminate()

if __name__ == "__main__":
    main()