import time
import numpy as np
import pyrr
import moderngl
import moderngl_window as mglw
from numba import njit
from scipy.special import hermite, factorial

@njit
def laplacian(array, dx2):
    Nx, Ny, Nz = array.shape
    lap = np.zeros_like(array)
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            for k in range(1, Nz-1):
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

def wavefunction(x, y, z, t, m=1.0, omega=1.0, n_x=2, n_y=2, n_z=2):
    hbar = 1.0  # natural units
    xi = np.sqrt(m * omega / hbar) * x
    yi = np.sqrt(m * omega / hbar) * y
    zi = np.sqrt(m * omega / hbar) * z
    norm = (m * omega / (np.pi * hbar))**0.5 / np.sqrt(2**(n_x+n_y+n_z) *
          factorial(n_x) * factorial(n_y) * factorial(n_z))
    H_nx = hermite(n_x)
    H_ny = hermite(n_y)
    H_nz = hermite(n_z)
    psi = norm * np.exp(-0.5 * (xi**2 + yi**2 + zi**2)) * H_nx(xi) * H_ny(yi) * H_nz(zi)
    return psi + 0j

def potential(x, y, z):
    scale_field_1 = 30
    narrowness_field_1 = 10
    field_1 = narrowness_field_1 * ((x/scale_field_1)**2 +
                                    (y/scale_field_1)**2 +
                                    (z/scale_field_1)**2)
    scale_field_2 = 3e5
    U0 = 1.0
    r0 = 20.0
    r_squared = x**2 + y**2 + z**2
    r_soft = np.sqrt(r_squared + r0**2)
    field_2 = U0 / (r_soft**3) * scale_field_2
    return field_1 + field_2

class SchrodingerSimulation3D(mglw.WindowConfig):
    title = "3D Volume Rendering of |ψ|² with Potential Overlay"
    window_size = (800, 800)
    gl_version = (3, 3)
    aspect_ratio = 1.0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_time = time.time()
        self.frame_count = 0

        # Grid parameters (using a cubic grid).
        self.N = 128  # Use a lower resolution for 3D textures for performance
        self.x = np.linspace(-50, 50, self.N)
        self.y = np.linspace(-50, 50, self.N)
        self.z = np.linspace(-50, 50, self.N)
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.dz = self.z[1] - self.z[0]
        self.t = 0.0

        # Initialize the wavefunction.
        self.psi = wavefunction(self.X, self.Y, self.Z, self.t)

        # Time evolution parameters.
        self.dt = 0.01
        self.num_repetitions = 1

        # Create a 3D texture for the probability density.
        density = np.abs(self.psi)**2
        norm = np.max(density) if np.max(density) != 0 else 1
        scaled = (density / norm * 255).astype(np.uint8)
        # The texture data needs shape (N, N, N, 3)
        volume_data = np.dstack([scaled] * 3).reshape((self.N, self.N, self.N, 3))
        self.volume_texture = self.ctx.texture3d((self.N, self.N, self.N), 3,
                                                 data=volume_data.tobytes())
        self.volume_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # Build a cube that spans the unit cube [0,1]^3.
        # Each vertex: 3D position and 3D texture coordinate.
        vertices = np.array([
            # positions         # texcoords
            0, 0, 0,           0, 0, 0,
            1, 0, 0,           1, 0, 0,
            1, 1, 0,           1, 1, 0,
            0, 1, 0,           0, 1, 0,
            0, 0, 1,           0, 0, 1,
            1, 0, 1,           1, 0, 1,
            1, 1, 1,           1, 1, 1,
            0, 1, 1,           0, 1, 1,
        ], dtype='f4')

        # Cube indices (two triangles per face, 6 faces).
        indices = np.array([
            0, 1, 2,  0, 2, 3,  # front
            1, 5, 6,  1, 6, 2,  # right
            5, 4, 7,  5, 7, 6,  # back
            4, 0, 3,  4, 3, 7,  # left
            3, 2, 6,  3, 6, 7,  # top
            4, 5, 1,  4, 1, 0   # bottom
        ], dtype='i4')

        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.ibo = self.ctx.buffer(indices.tobytes())

        # Volume rendering shader (using a simple ray marching approach)
        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec3 in_vert;
                in vec3 in_tex;
                out vec3 v_tex;
                uniform mat4 mvp;
                void main() {
                    // Pass the texture coordinates unchanged.
                    v_tex = in_tex;
                    gl_Position = mvp * vec4(in_vert, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                uniform sampler3D volumeTexture;
                in vec3 v_tex;
                out vec4 f_color;

                // A basic transfer function: map density to a grayscale color.
                vec4 transferFunction(float density) {
                    return vec4(density, density, density, density);
                }

                void main(){
                    // Ray marching setup:
                    vec3 rayDir = normalize(v_tex - vec3(0.5));
                    vec3 rayOrigin = v_tex;
                    float stepSize = 0.01;
                    vec4 colorAccum = vec4(0.0);
                    // March along the ray within the cube.
                    for (float t = 0.0; t < 1.0; t += stepSize) {
                        vec3 pos = rayOrigin + t * rayDir;
                        // Check if pos is within the cube [0,1].
                        if (any(lessThan(pos, vec3(0.0))) || any(greaterThan(pos, vec3(1.0))))
                            continue;
                        float d = texture(volumeTexture, pos).r;
                        vec4 col = transferFunction(d);
                        // Front-to-back compositing.
                        colorAccum.rgb += (1.0 - colorAccum.a) * col.rgb * col.a;
                        colorAccum.a += (1.0 - colorAccum.a) * col.a;
                        if (colorAccum.a >= 0.95)
                            break;
                    }
                    f_color = colorAccum;
                }
            ''',
        )
        # We'll use a simple MVP matrix. Here, we define a basic perspective view.
        # For a more complete implementation, you might add controls to rotate/zoom.
        self.mvp = self.prog['mvp']
        self.camera_matrix = self.create_perspective()

        # Create the vertex array object.
        self.vao = self.ctx.vertex_array(
            self.prog,
            [(self.vbo, '3f 3f', 'in_vert', 'in_tex')],
            self.ibo
        )
        

    def create_perspective(self):
        eye = np.array([1.5, 1.5, 2.0])
        target = np.array([0.5, 0.5, 0.5])
        up = np.array([0.0, 1.0, 0.0])
        view = pyrr.matrix44.create_look_at(eye, target, up)
        proj = pyrr.matrix44.create_perspective_projection(45.0, self.aspect_ratio, 0.1, 10.0)
        mvp = pyrr.matrix44.multiply(proj, view)  # Note the order: proj * view
        return mvp.astype('f4')

    def update(self, time_delta):
        # Evolve the wavefunction (using RK4 steps).
        for _ in range(self.num_repetitions):
            k1 = self.dt * time_dependent_3D_schrodinger(self.psi, potential,
                                                          self.X, self.Y, self.Z,
                                                          self.dx, self.dy, self.dz)
            k2 = self.dt * time_dependent_3D_schrodinger(self.psi + k1 / 2, potential,
                                                          self.X, self.Y, self.Z,
                                                          self.dx, self.dy, self.dz)
            k3 = self.dt * time_dependent_3D_schrodinger(self.psi + k2 / 2, potential,
                                                          self.X, self.Y, self.Z,
                                                          self.dx, self.dy, self.dz)
            k4 = self.dt * time_dependent_3D_schrodinger(self.psi + k3, potential,
                                                          self.X, self.Y, self.Z,
                                                          self.dx, self.dy, self.dz)
            self.psi += (k1 + 2 * k2 + 2 * k3 + k4) / 6

        # Update the 3D texture.
        density = np.abs(self.psi)**2
        norm = np.max(density) if np.max(density) != 0 else 1
        scaled = (density / norm * 255).astype(np.uint8)
        volume_data = np.dstack([scaled] * 3).reshape((self.N, self.N, self.N, 3))
        self.volume_texture.write(volume_data.tobytes())

        self.frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.last_time
        if elapsed_time >= 1.0:
            fps = self.frame_count / elapsed_time
            self.wnd.title = f"3D Volume Rendering - FPS: {fps:.2f}"
            self.last_time = current_time
            self.frame_count = 0

    def on_render(self, time, frame_time):
        self.update(frame_time)
        self.ctx.clear(0.0, 0.0, 0.0)
        # Update the MVP uniform.
        self.mvp.write(self.camera_matrix.tobytes())
        self.volume_texture.use(location=0)
        self.vao.render()

if __name__ == '__main__':
    mglw.run_window_config(SchrodingerSimulation3D)
