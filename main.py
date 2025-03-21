import time
import numpy as np
import moderngl
import moderngl_window as mglw
from numba import njit

@njit
def laplacian(array, dx2):
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
    lap = laplacian(psi, dx2)
    rhs = -hbar**2/(2*mass) * lap + potential(x, y) * psi
    dpsi_dt = rhs / (hbar * 1j)
    return dpsi_dt

def wavefunction(x, y, t, sigma=1.0, k_x=1.0, k_y=1.0, omega=1.0):
    A = 1.0 / ((2 * np.pi * sigma**2)**0.5)

    return A * np.exp(-((x**2 + y**2) / (2 * sigma**2))) * np.exp(1j*(k_x*x + k_y*y - omega*t))
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
    
class SchrodingerSimulation(mglw.WindowConfig):
    title = "Probability Density |ψ|² with Potential Overlay"
    window_size = (800, 800)
    gl_version = (3, 3)
    aspect_ratio = 1.0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        #FPS calculation
        self.last_time = time.time() 
        self.frame_count = 0

        # Define grid parameters.
        self.N = 500
        self.x = np.linspace(-50, 50, self.N)
        self.y = np.linspace(-50, 50, self.N)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.t = 0.0

        # Initialize the wavefunction.
        self.psi = wavefunction(self.X, self.Y, self.t, sigma=2.0)

        # Time evolution parameters.
        self.dt = 0.01
        self.num_repetitions = 25

        # Create dynamic probability density texture.
        density = np.abs(self.psi)**2
        norm = np.max(density) if np.max(density) != 0 else 1
        scaled = (density / norm * 255).astype(np.uint8)
        self.image_data = np.dstack([scaled] * 3)
        self.texture = self.ctx.texture((self.N, self.N), 3, data=self.image_data.tobytes())
        self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # Create static potential energy texture.
        potential_field = potential(self.X, self.Y)
        p_min = potential_field.min()
        p_max = potential_field.max()
        norm_potential = (potential_field - p_min) / (p_max - p_min)
        p_scaled = (norm_potential * 255).astype(np.uint8)
        p_image = np.dstack([p_scaled] * 3)
        self.potential_texture = self.ctx.texture((self.N, self.N), 3, data=p_image.tobytes())
        self.potential_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # Shader program with two textures.
        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec2 in_vert;
                in vec2 in_tex;
                out vec2 v_tex;
                void main() {
                    gl_Position = vec4(in_vert, 0.0, 1.0);
                    v_tex = in_tex;
                }
            ''',
            fragment_shader='''
                #version 330
                uniform sampler2D Texture;           // probability density texture
                uniform sampler2D PotentialTexture;    // potential energy texture
                in vec2 v_tex;
                out vec4 f_color;

                // Bone colormap for probability density.
                vec3 bone(float t) {
                    vec3 lowColor = vec3(0.0, 0.0, 0.12);
                    vec3 highColor = vec3(1.0, 0.95, 0.88);
                    return mix(lowColor, highColor, t);
                }
                // Potential energy colormap: gradient from dark purple to light blue.
                vec3 potentialColor(float t) {
                    vec3 low = vec3(0.0, 0.0, 0.0);
                    vec3 high = vec3(0.2, 0.8, 1.0);
                    return mix(low, high, t);
                }
                void main() {
                    float pd = texture(Texture, v_tex).r;          // intensity from probability density
                    float pe = texture(PotentialTexture, v_tex).r;   // intensity from potential energy
                    vec3 pd_color = bone(pd);
                    vec3 pe_color = potentialColor(pe);
                    float overlay_alpha = 0.3; // blend factor for potential overlay
                    vec3 final_color = mix(pd_color, pe_color, overlay_alpha);
                    f_color = vec4(final_color, 1.0);
                }
            ''',
        )

        # Set up a full-screen quad.
        vertices = np.array([
            -1.0, -1.0, 0.0, 0.0,
             1.0, -1.0, 1.0, 0.0,
            -1.0,  1.0, 0.0, 1.0,
             1.0,  1.0, 1.0, 1.0,
        ], dtype='f4')

        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.vao = self.ctx.vertex_array(
            self.prog,
            [(self.vbo, '2f 2f', 'in_vert', 'in_tex')]
        )

    def update(self, time_delta):
        # Evolve the wavefunction with RK4 steps.
        for _ in range(self.num_repetitions):
            k1 = self.dt * time_dependent_2D_schrodinger(self.psi, potential, self.X, self.Y, self.dx, self.dy)
            k2 = self.dt * time_dependent_2D_schrodinger(self.psi + k1 / 2, potential, self.X, self.Y, self.dx, self.dy)
            k3 = self.dt * time_dependent_2D_schrodinger(self.psi + k2 / 2, potential, self.X, self.Y, self.dx, self.dy)
            k4 = self.dt * time_dependent_2D_schrodinger(self.psi + k3, potential, self.X, self.Y, self.dx, self.dy)
            self.psi += (k1 + 2 * k2 + 2 * k3 + k4) / 6

        # Update the probability density texture.
        density = np.abs(self.psi)**2
        norm = np.max(density) if np.max(density) != 0 else 1
        scaled = (density / norm * 255).astype(np.uint8)
        image_data = np.dstack([scaled] * 3)
        self.texture.write(image_data.tobytes())
        
        # Calculate FPS
        self.frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.last_time

        if elapsed_time >= 1.0:  # Update every second
            fps = self.frame_count / elapsed_time
            self.wnd.title = f"ModernGL Window - FPS: {fps:.2f}"
            self.last_time = current_time
            self.frame_count = 0
            
            print(fps)

    def on_render(self, time, frame_time):
        self.update(frame_time)
        self.ctx.clear(0.0, 0.0, 0.0)
        # Bind textures to their respective texture units.
        self.texture.use(location=0)
        self.potential_texture.use(location=1)
        self.prog['Texture'] = 0
        self.prog['PotentialTexture'] = 1
        self.vao.render(moderngl.TRIANGLE_STRIP)
    

if __name__ == '__main__':
    mglw.run_window_config(SchrodingerSimulation)