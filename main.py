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
# Helper: Electron Quantum Numbers
# ---------------------------
def electron_quantum_numbers(atomic_number):
    orbitals = [
        (1, 0),  # 1s
        (2, 0),  # 2s
        (2, 1),  # 2p
        (3, 0),  # 3s
        (3, 1),  # 3p
        (4, 0),  # 4s
        (3, 2),  # 3d
        (4, 1),  # 4p
        (5, 0),  # 5s
        (4, 2),  # 4d
        (5, 1),  # 5p
        (6, 0),  # 6s
        (4, 3),  # 4f  (lanthanides)
        (5, 2),  # 5d
        (6, 1),  # 6p
        (7, 0),  # 7s
        (5, 3),  # 5f  (actinides)
        (6, 2),  # 6d
        (7, 1)   # 7p
    ]
    orbital_capacity = {0: 2, 1: 6, 2: 10, 3: 14}
    electrons_remaining = atomic_number
    electrons_list = []
    for n, l in orbitals:
        if electrons_remaining <= 0:
            break
        capacity = orbital_capacity[l]
        electrons_in_orbital = min(capacity, electrons_remaining)
        electrons_remaining -= electrons_in_orbital
        m_values = list(range(-l, l + 1))
        first_pass = []
        second_pass = []
        for m in m_values:
            if len(first_pass) < electrons_in_orbital:
                first_pass.append((n, l, m))
            else:
                break
        for m in m_values:
            if len(first_pass) + len(second_pass) < electrons_in_orbital:
                second_pass.append((n, l, m))
            else:
                break
        electrons_list.extend(first_pass + second_pass)
    return electrons_list

# ---------------------------
# Global Simulation Parameters
# ---------------------------
num_r = 50         # radial grid points
num_theta = 100    # angular grid (theta)
num_phi = 100      # angular grid (phi)

R_box = 10.0
rcut_percent = 0.9  # R_cut = rcut_percent * R_box

wavefunction_options = ["hydrogen", "gaussian", "oscillator", "vortex", "atom", "sinc", "bessel", "sech", "hermite"]
current_wavefunction_index = 0
wavefunction_type = wavefunction_options[current_wavefunction_index]

evolution_options = ["static", "phase", "full"]
current_evolution_index = 0  # default "static"
evolution_mode = evolution_options[current_evolution_index]

slow_factor = 1.0

# For Hydrogen
n_val = 2
l_val = 1
m_val = 1
a0_val = 0.5

# For Atom (composite) wavefunction
atomic_number = 61
orbitals = electron_quantum_numbers(atomic_number)
atom_a0 = 0.5
deltaHF = 0.1
U_coulomb = 0.05
U_exchange = 0.02
spin = 1.0


# For Gaussian
x0_vals = [0.0, 0.0, 0.0]
sigma_val = 1.0
p0_vals = [10.0, 0.0, 0.0]

# For Oscillator
oscillator_omega = 0.5  # Angular frequency

# For Vortex
vortex_sigma = 1.0
vortex_energy = 1.0

# For Sinc wavefunction
sinc_k0 = 1.0

# For Bessel wavefunction:
bessel_kr = 1.0  # radial wave number
bessel_kz = 1.0  # axial wave number

# For sech wavefunction
sech_a = 1.0  # controls the width of the hyperbolic secant profile
sech_energy = 1.0

# New parameters for the Hermite–Gaussian wavefunction:
hg_width = 1.0       # Beam waist (spatial scaling)
hg_order_m = 1       # Order in x-direction
hg_order_n = 1       # Order in y-direction

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
    
    // --- Utility Functions ---
    float factorial(int k) {
        float result = 1.0;
        for (int i = 1; i <= k; i++) {
            result *= float(i);
        }
        return result;
    }
    
    float assocLaguerre(int k, int alpha, float x) {
        if (k == 0) return 1.0;
        if (k == 1) return float(alpha + 1) - x;
        float Lkm2 = 1.0;
        float Lkm1 = float(alpha + 1) - x;
        float Lk = 0.0;
        for (int i = 2; i <= k; i++) {
            Lk = ((2.0 * float(i) + float(alpha) - 1.0 - x) * Lkm1 - (float(i + alpha - 1)) * Lkm2) / float(i);
            Lkm2 = Lkm1;
            Lkm1 = Lk;
        }
        return Lk;
    }
    
    float assocLegendre(int l, int m, float x) {
        float pmm = 1.0;
        if (m > 0) {
            float somx2 = sqrt(1.0 - x * x);
            float fact = 1.0;
            for (int i = 1; i <= m; i++) {
                pmm *= -fact * somx2;
                fact += 2.0;
            }
        }
        if (l == m) return pmm;
        float pmmp1 = x * (2.0 * float(m) + 1.0) * pmm;
        if (l == m + 1) return pmmp1;
        float pll = 0.0;
        for (int ll = m + 2; ll <= l; ll++) {
            pll = ((2.0 * float(ll) - 1.0) * x * pmmp1 - (float(ll + m - 1))) * pmm / float(ll - m);
            pmm = pmmp1;
            pmmp1 = pll;
        }
        return pll;
    }
    
    // --- Hydrogen Kernel (for one orbital) ---
    float2 computeHydrogen(float r, float theta, float phi,
                        float n, float l, float m, float a0) {
        float rho = 2.0 * r / (n * a0);
        float norm_radial = sqrt(pow(2.0 / (n * a0), 3.0) *
                        (factorial(int(n - l - 1)) / (2.0 * n * factorial(int(n + l)))));
        float radial = norm_radial * exp(-r/(n * a0)) * pow(rho, l);
        int k = int(n - l - 1);
        float laguerre = assocLaguerre(k, int(2 * l + 1), rho);
        radial *= laguerre;
        
        float norm_angular = sqrt((2.0 * l + 1.0) / (4.0 * 3.141592653589793) *
                        (factorial(int(l - abs(int(m)))) / factorial(int(l + abs(int(m))))));
        float legendre = assocLegendre(int(l), int(abs(m)), cos(theta));
        float angular = norm_angular * legendre;
        
        float psi = radial * angular;
        float psi_real = psi * cos(m * phi);
        float psi_imag = psi * sin(m * phi);
        return float2(psi_real, psi_imag);
    }
    
    // --- SCF Composite Atom Kernel with Coulomb and Exchange (LDA-based) ---
    kernel void computeCompositeAtomWavefunction(const device float* inCoords [[ buffer(0) ]],
                                                      device float2* outWF [[ buffer(1) ]],
                                                      constant float* params [[ buffer(2) ]],
                                                      uint id [[ thread_position_in_grid ]])
    {
        // --- Global Parameters ---
        // params[0] : numOrbitals
        // params[1] : a0 (Bohr radius scale)
        // params[2] : deltaHF (global energy shift)
        // params[3] : U_coulomb (Coulomb interaction strength)
        // params[4] : U_exchange (exchange scaling; used to compute local exchange potential)
        // params[5] : t (time)
        uint numOrbitals = uint(params[0]);
        float a0 = params[1];
        float deltaHF = params[2];
        float U_coulomb = params[3];
        float U_exchange = params[4];
        float t = params[5];
        
        // --- Get spatial coordinates for this thread: (r, theta, phi) ---
        uint index = id * 3;
        float r = inCoords[index];
        float theta = inCoords[index + 1];
        float phi = inCoords[index + 2];
        
        // --- First pass: compute local electron density and spin density ---
        float density = 0.0;
        float spinDensity = 0.0;
        // Also store the individual orbital values for reuse.
        thread float2 psiOrbitals[64]; // adjust if necessary for maximum orbitals
        
        for (uint i = 0; i < numOrbitals; i++) {
            uint base = 6 + i * 4; // starting after global parameters (6 floats)
            float n = params[base + 0];
            float l = params[base + 1];
            float m = params[base + 2];
            float spin = params[base + 3];   // +1 or -1
            
            float2 psi = computeHydrogen(r, theta, phi, n, l, m, a0);
            psiOrbitals[i] = psi;
            
            float amp2 = psi.x * psi.x + psi.y * psi.y;
            density += amp2;
            spinDensity += spin * amp2;
        }
        
        // --- Derive local potentials from the density ---
        // Coulomb potential: proportional to local electron density.
        float V_coulomb = U_coulomb * density;
        
        // Exchange (Fock) potential: using a simple LDA approximation (based on Slater exchange).
        float V_exchange = 0.0;
        if (density > 1e-6) {
            // The factor (3/(π))^(1/3) is typical in Slater's exchange formula.
            float slaterFactor = pow(3.0 / 3.141592653589793, 1.0/3.0);
            // The sign of the exchange is modulated by net spin polarization.
            V_exchange = -U_exchange * slaterFactor * pow(density, 1.0/3.0) * ((spinDensity >= 0.0) ? 1.0 : -1.0);
        }
        
        // --- Compute Composite Wavefunction ---
        float2 psi_sum = float2(0.0, 0.0);
        for (uint i = 0; i < numOrbitals; i++) {
            uint base = 6 + i * 4;
            float n = params[base + 0];
            float l = params[base + 1];
            float m = params[base + 2];
            // Spin is already used in the density; not further used here directly.
            
            float2 psi = psiOrbitals[i];
            // Hydrogen-like base energy
            float E_base = -1.0 / (2.0 * n * n);
            // Effective energy includes the base energy, the SCF mean-field contributions, and the global shift.
            float E_effective = E_base + deltaHF + V_coulomb + V_exchange;
            float phase = E_effective * t;
            float cosPhase = cos(phase);
            float sinPhase = sin(phase);
            float2 psi_evolved = float2(psi.x * cosPhase - psi.y * sinPhase,
                                        psi.x * sinPhase + psi.y * cosPhase);
            psi_sum += psi_evolved;
        }
        
        // Write the composite wavefunction back to the buffer.
        outWF[id] = psi_sum;
    }
        
    // --- Other Kernels ---
    kernel void computeHydrogenWavefunction(const device float* inCoords [[ buffer(0) ]],
                                            device float2* outWF [[ buffer(1) ]],
                                            constant float4& params [[ buffer(2) ]],
                                            uint id [[ thread_position_in_grid ]]) {
        uint index = id * 3;
        float r = inCoords[index];
        float theta = inCoords[index + 1];
        float phi = inCoords[index + 2];
        
        float n = params.x;
        float l = params.y;
        float m = params.z;
        float a0 = params.w;
        
        float norm_radial = sqrt(pow(2.0 / (n * a0), 3.0) *
                        (factorial(int(n - l - 1)) / (2.0 * n * factorial(int(n + l)))));
        float rho = 2.0 * r / (n * a0);
        float radial = norm_radial * exp(-r/(n * a0)) * pow(rho, l);
        int k = int(n - l - 1);
        float laguerre = assocLaguerre(k, int(2 * l + 1), rho);
        radial *= laguerre;
        
        float norm_angular = sqrt((2.0 * l + 1.0) / (4.0 * 3.141592653589793) *
                        (factorial(int(l - abs(int(m)))) / factorial(int(l + abs(int(m))))));
        float legendre = assocLegendre(int(l), int(abs(m)), cos(theta));
        float angular = norm_angular * legendre;
        
        float psi = radial * angular;
        float psi_real = psi * cos(m * phi);
        float psi_imag = psi * sin(m * phi);
        
        outWF[id] = float2(psi_real, psi_imag);
    }
    
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
        
    kernel void computeOscillatorWavefunction(const device float* inCoords [[ buffer(0) ]],
                                                device float2* outWF [[ buffer(1) ]],
                                                constant float& params [[ buffer(2) ]],
                                                uint id [[ thread_position_in_grid ]]) {
        uint index = id * 3;
        float r = inCoords[index];
        float psi = exp(-params * r * r / 2.0);
        float norm = pow(params/3.141592653589793, 0.75);
        psi *= norm;
        outWF[id] = float2(psi, 0.0);
    }
        
    kernel void computeVortexWavefunction(const device float* inCoords [[ buffer(0) ]],
                                            device float2* outWF [[ buffer(1) ]],
                                            constant float& params [[ buffer(2) ]],
                                            uint id [[ thread_position_in_grid ]]) {
        uint index = id * 3;
        float r = inCoords[index];
        float theta = inCoords[index + 1];
        float phi = inCoords[index + 2];
        float amplitude = r * sin(theta) * exp(-r*r/(2.0 * params * params));
        float A = pow(1.0/(3.141592653589793 * params * params), 0.75);
        amplitude *= A;
        float psi_real = amplitude * cos(phi);
        float psi_imag = amplitude * sin(phi);
        outWF[id] = float2(psi_real, psi_imag);
    }
    
    // --- New Sinc Kernel ---
    kernel void computeSincWavefunction(const device float* inCoords [[ buffer(0) ]],
                                          device float2* outWF [[ buffer(1) ]],
                                          constant float& params [[ buffer(2) ]],
                                          uint id [[ thread_position_in_grid ]])
    {
        uint index = id * 3;
        float r = inCoords[index];
        float k0 = params;
        float psi = (r == 0.0) ? 1.0 : sin(k0 * r) / (k0 * r);
        outWF[id] = float2(psi, 0.0);
    }
    
    // --- New Bessel Kernel ---
    // Helper: Approximate cylindrical Bessel function J0 using a low-order expansion for small x
    float besselJ0(float x) {
        if (x < 3.0) {
            // 4th order Taylor expansion for small x: J0(x) ~ 1 - (x^2)/4 + (x^4)/64
            return 1.0 - 0.25 * x * x + (1.0/64.0) * x * x * x * x;
        } else {
            // Asymptotic form for large x: J0(x) ~ sqrt(2/(pi*x))*cos(x - pi/4)
            return sqrt(2.0/(3.141592653589793*x)) * cos(x - 3.141592653589793/4.0);
        }
    }
    
    kernel void computeBesselWavefunction(const device float* inCoords [[ buffer(0) ]],
                                            device float2* outWF [[ buffer(1) ]],
                                            constant float2& params [[ buffer(2) ]],
                                            uint id [[ thread_position_in_grid ]])
    {
        uint index = id * 3;
        float r = inCoords[index];
        float theta = inCoords[index + 1];
        float phi = inCoords[index + 2];
        
        // Convert spherical (r, theta, phi) to cylindrical coordinates:
        float rho = r * sin(theta);
        float z = r * cos(theta);
        
        float k_r = params.x;
        float k_z = params.y;
        
        // Compute amplitude via the approximated Bessel J0 function:
        float amplitude = besselJ0(k_r * rho);
        // Add a phase factor along z:
        float phase = k_z * z;
        
        float psi_real = amplitude * cos(phase);
        float psi_imag = amplitude * sin(phase);
        outWF[id] = float2(psi_real, psi_imag);
    }
    
    kernel void computeSechWavefunction(uint id [[thread_position_in_grid]],
                                          constant float* spherical_coords [[buffer(0)]],
                                          device float* out_data [[buffer(1)]],
                                          constant float* params [[buffer(2)]])
    {
        // Extract the parameter controlling the width.
        float sech_a = params[0];
        
        // The radial coordinate (assumed to be the first of three floats per point).
        float r = spherical_coords[id * 3];
        
        // Compute the real part as 1/cosh(r/sech_a) and set the imaginary part to zero.
        float psi_real = 1.0 / cosh(r / sech_a);
        float psi_imag = 0.0;
        
        out_data[id * 2] = psi_real;
        out_data[id * 2 + 1] = psi_imag;
    }
    
    // Compute the nth Hermite polynomial using a simple recurrence.
    float hermitePolynomial(int n, float x) {
        if (n == 0) return 1.0;
        if (n == 1) return 2.0 * x;
        float Hn_2 = 1.0;
        float Hn_1 = 2.0 * x;
        float Hn = 0.0;
        for (int i = 2; i <= n; i++) {
             Hn = 2.0 * x * Hn_1 - 2.0 * float(i - 1) * Hn_2;
             Hn_2 = Hn_1;
             Hn_1 = Hn;
        }
        return Hn;
    }
    
    kernel void computeHermiteGaussianWavefunction(uint id [[thread_position_in_grid]],
                                                   constant float* spherical_coords [[buffer(0)]],
                                                   device float* out_data [[buffer(1)]],
                                                   constant float* params [[buffer(2)]])
    {
        // Read in parameters: beam waist and orders.
        float hg_width = params[0];
        int hg_order_m = int(params[1]);
        int hg_order_n = int(params[2]);
        
        // Retrieve spherical coordinates: assume layout [r, theta, phi, ...]
        float r = spherical_coords[id * 3];
        float theta = spherical_coords[id * 3 + 1];
        float phi = spherical_coords[id * 3 + 2];
        
        // Convert to Cartesian coordinates (for x and y).
        float x = r * sin(phi) * cos(theta);
        float y = r * sin(phi) * sin(theta);
        
        // Normalize the coordinates for the Hermite polynomial evaluation.
        float xn = sqrt(2.0) * x / hg_width;
        float yn = sqrt(2.0) * y / hg_width;
        
        // Calculate Hermite polynomials for orders hg_order_m and hg_order_n.
        float Hm = hermitePolynomial(hg_order_m, xn);
        float Hn = hermitePolynomial(hg_order_n, yn);
        
        // Hermite–Gaussian amplitude: product of the two Hermite polynomials and a Gaussian envelope.
        float amplitude = Hm * Hn * exp(-(x*x + y*y) / (hg_width * hg_width));
        
        // For simplicity, assign zero to the imaginary part.
        out_data[id * 2] = amplitude;
        out_data[id * 2 + 1] = 0.0;
    }
'''

# ---------------------------
# Precompute Wavefunctions
# ---------------------------
def precompute_wavefunctions(spherical_coords):
    global n_val, l_val, m_val, a0_val, x0_vals, sigma_val, p0_vals, wavefunction_type, oscillator_omega, vortex_sigma, orbitals, atom_a0, sinc_k0
    num_points = spherical_coords.shape[0]
    in_data = spherical_coords.astype(np.float32).ravel().tobytes()
    dev = mc.Device()
    in_buf = dev.buffer(in_data)
    out_size = num_points * 2 * 4  # 2 floats per point
    out_buf = dev.buffer(out_size)
    if wavefunction_type == "hydrogen":
        params = np.array([float(n_val), float(l_val), float(m_val), a0_val],
                          dtype=np.float32).tobytes()
        param_buf = dev.buffer(params)
        kernel_fn = dev.kernel(metal_shader).function("computeHydrogenWavefunction")
    elif wavefunction_type == "gaussian":
        params = np.array([x0_vals[0], x0_vals[1], x0_vals[2],
                           sigma_val, p0_vals[0], p0_vals[1], p0_vals[2]],
                          dtype=np.float32).tobytes()
        param_buf = dev.buffer(params)
        kernel_fn = dev.kernel(metal_shader).function("computeGaussianWavepacket")
    elif wavefunction_type == "oscillator":
        params = np.array([oscillator_omega], dtype=np.float32).tobytes()
        param_buf = dev.buffer(params)
        kernel_fn = dev.kernel(metal_shader).function("computeOscillatorWavefunction")
    elif wavefunction_type == "vortex":
        params = np.array([vortex_sigma], dtype=np.float32).tobytes()
        param_buf = dev.buffer(params)
        kernel_fn = dev.kernel(metal_shader).function("computeVortexWavefunction")
    elif wavefunction_type == "atom":
        num_orbitals = len(orbitals)
        initial_time = 0.0
        params_list = [float(num_orbitals), atom_a0, deltaHF, U_coulomb, U_exchange, initial_time]
        # Append each orbital's [n, l, m, spin]. Here we set spin to +1.0 for each orbital.
        for orb in orbitals:
            params_list.extend([float(orb[0]), float(orb[1]), float(orb[2]), spin])
        params = np.array(params_list, dtype=np.float32).tobytes()
        param_buf = dev.buffer(params)
        kernel_fn = dev.kernel(metal_shader).function("computeCompositeAtomWavefunction")
    elif wavefunction_type == "sinc":
        params = np.array([sinc_k0], dtype=np.float32).tobytes()
        param_buf = dev.buffer(params)
        kernel_fn = dev.kernel(metal_shader).function("computeSincWavefunction")
    elif wavefunction_type == "bessel":
        # For the bessel kernel, we need two parameters: k_r and k_z.
        params = np.array([bessel_kr, bessel_kz], dtype=np.float32).tobytes()
        param_buf = dev.buffer(params)
        kernel_fn = dev.kernel(metal_shader).function("computeBesselWavefunction")
    elif wavefunction_type == "sech":
        # New "sech" wavefunction: psi = 1/cosh(r/sech_a)
        params = np.array([sech_a], dtype=np.float32).tobytes()
        param_buf = dev.buffer(params)
        kernel_fn = dev.kernel(metal_shader).function("computeSechWavefunction")
    elif wavefunction_type == "hermite":
        # New Hermite–Gaussian wavefunction parameters: hg_width, hg_order_m, hg_order_n.
        params = np.array([hg_width, float(hg_order_m), float(hg_order_n)], dtype=np.float32).tobytes()
        param_buf = dev.buffer(params)
        kernel_fn = dev.kernel(metal_shader).function("computeHermiteGaussianWavefunction")
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
    global wavefunction_type, evolution_mode, n_val, l_val, m_val, a0_val, x0_vals, sigma_val, p0_vals
    global atomic_number, orbitals, atom_a0, oscillator_omega, vortex_sigma, vortex_energy, sinc_k0
    global bessel_kr, bessel_kz, sech_a, sech_energy, hg_width, hg_order_m, hg_order_n, deltaHF, U_coulomb, U_exchange, spin
    global R_box, rcut_percent, slow_factor
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
        
        if wavefunction_type == "hydrogen":
            changed, n_val = imgui.input_int("n", n_val)
            changed, l_val = imgui.input_int("l", l_val)
            changed, m_val = imgui.input_int("m", m_val)
            changed, a0_val = imgui.input_float("a0", a0_val)
        elif wavefunction_type == "gaussian":
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
        elif wavefunction_type == "oscillator":
            changed, oscillator_omega = imgui.input_float("omega", oscillator_omega)
        elif wavefunction_type == "vortex":
            changed, vortex_sigma = imgui.input_float("vortex sigma", vortex_sigma)
            changed, vortex_energy = imgui.input_float("vortex energy", vortex_energy)
        elif wavefunction_type == "atom":
            changed, atomic_number = imgui.input_int("atomic number", atomic_number)
            if changed:
                orbitals = electron_quantum_numbers(atomic_number)
            changed, atom_a0 = imgui.input_float("atom a0", atom_a0)
            changed, deltaHF = imgui.input_float("HF Correction Value", deltaHF)
            changed, U_coulomb = imgui.input_float("Coulomb Parameter", U_coulomb)
            changed, U_exchange = imgui.input_float("Exchange Parameter", U_exchange)
            changed, spin = imgui.input_float("spin", spin)
        elif wavefunction_type == "sinc":
            changed, sinc_k0 = imgui.input_float("sinc k0", sinc_k0)
        elif wavefunction_type == "bessel":
            changed, bessel_kr = imgui.input_float("bessel k_r", bessel_kr)
            changed, bessel_kz = imgui.input_float("bessel k_z", bessel_kz)
        elif wavefunction_type == "sech":
            changed, sech_a = imgui.input_float("sech width", sech_a)
            changed, sech_energy = imgui.input_float("sech energy", sech_energy)
        elif wavefunction_type == "hermite":
            changed, hg_width = imgui.input_float("HG width", hg_width)
            changed, hg_order_m = imgui.input_int("HG order m", hg_order_m)
            changed, hg_order_n = imgui.input_int("HG order n", hg_order_n)


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
            if wavefunction_type == "hydrogen":
                E = -1.0 / (2.0 * n_val * n_val)
                lam = (1 - 1j*dt*E/2) / (1 + 1j*dt*E/2)
                psi_complex *= lam
            elif wavefunction_type == "oscillator":
                E = 1.5 * oscillator_omega
                lam = (1 - 1j*dt*E/2) / (1 + 1j*dt*E/2)
                psi_complex *= lam
            elif wavefunction_type == "gaussian":
                E = 1.0
                lam = (1 - 1j*dt*E/2) / (1 + 1j*dt*E/2)
                psi_complex *= lam
            elif wavefunction_type == "vortex":
                E = vortex_energy
                lam = (1 - 1j*dt*E/2) / (1 + 1j*dt*E/2)
                psi_complex *= lam
            elif wavefunction_type == "atom":
                atom_time += dt
                E = -1.0 / (2 * orbitals[0][0] * orbitals[0][0])
                precomp = precomputed_wf[:,0] + 1j * precomputed_wf[:,1]
                psi_complex = precomp * np.exp(-1j * E * atom_time)
            elif wavefunction_type == "sinc":
                E = sinc_k0**2 / 2.0
                lam = (1 - 1j*dt*E/2) / (1 + 1j*dt*E/2)
                psi_complex *= lam
            elif wavefunction_type == "bessel":
                # For a Bessel beam, assume dispersion relation: E = (k_r^2 + k_z^2)/2 (in atomic units)
                E = (bessel_kr**2 + bessel_kz**2) / 2.0
                lam = (1 - 1j*dt*E/2) / (1 + 1j*dt*E/2)
                psi_complex *= lam
            elif wavefunction_type == "sech":
                # Add a simple phase evolution. You might want to set an energy value or derive one from sech_a.
                E = sech_energy  # choose an appropriate energy scale
                lam = (1 - 1j*dt*E/2) / (1 + 1j*dt*E/2)
                psi_complex *= lam
            elif wavefunction_type == "hermite":
                # Treat the Hermite–Gaussian mode as an eigenstate of a harmonic oscillator.
                # A 2D oscillator has energy levels proportional to (m+n+1).
                E = (hg_order_m + hg_order_n + 1.0)
                lam = (1 - 1j * dt * E/2) / (1 + 1j * dt * E/2)
                psi_complex *= lam
            psi_flat = psi_complex.ravel()
        elif evolution_mode == "full":
            # For full evolution we update spatially using finite differences.
            if wavefunction_type == "hydrogen":
                # Now update hydrogen spatially using a Coulomb potential.
                dt_eff = dt * slow_factor
                psi_complex = psi_complex.reshape((num_r, num_theta*num_phi))
                dr = r_vals[1] - r_vals[0]
                N = num_r - 1  # interior radial grid points; enforce psi[N]=0.
                r_interior = r_vals[:N]
                epsilon = 1e-3
                V = -1.0/(r_interior + epsilon)   # Coulomb potential (regularized)
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
                psi_updated[N, :] = 0.0  # enforce Dirichlet boundary
                psi_complex = psi_updated
                psi_flat = psi_complex.ravel()
            elif wavefunction_type in ["oscillator", "gaussian", "vortex"]:
                dt_eff = dt * slow_factor
                psi_complex = psi_complex.reshape((num_r, num_theta*num_phi))
                dr = r_vals[1] - r_vals[0]
                N = num_r - 1
                r_interior = r_vals[:N]
                if wavefunction_type == "oscillator":
                    V = 0.5 * oscillator_omega**2 * r_interior**2
                elif wavefunction_type == "gaussian":
                    V = np.where(r_interior < R_cut, 0.0, V0)
                elif wavefunction_type == "vortex":
                    V = np.zeros_like(r_interior)
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
            elif wavefunction_type == "atom":
                dt_eff = dt * slow_factor
                psi_complex = psi_complex.reshape((num_r, num_theta*num_phi))
                dr = r_vals[1] - r_vals[0]
                N = num_r - 1
                r_interior = r_vals[:N]
                epsilon = 1e-3
                V = -1.0/(r_interior+epsilon)
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
            elif wavefunction_type == "sinc":
                dt_eff = dt * slow_factor
                psi_complex = psi_complex.reshape((num_r, num_theta*num_phi))
                dr = r_vals[1] - r_vals[0]
                N = num_r - 1
                r_interior = r_vals[:N]
                V = np.zeros_like(r_interior)
                a = np.zeros(N-1, dtype=float)
                b = np.zeros(N, dtype=float)
                c = np.zeros(N-1, dtype=float)
                b[0] = -1.0/(dr**2)
                c[0] = 1.0/(dr**2)
                for i in range(1, N):
                    a[i-1] = -1.0/(2*dr**2) + 1.0/(4*r_vals[i]*dr)
                    b[i] = 1.0/(dr**2)
                    if i < N:
                        c[i-1] = -1.0/(2*dr**2) - 1.0/(4*r_vals[i]*dr)
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
            elif wavefunction_type == "bessel":
                dt_eff = dt * slow_factor
                psi_complex = psi_complex.reshape((num_r, num_theta*num_phi))
                dr = r_vals[1] - r_vals[0]
                N = num_r - 1
                r_interior = r_vals[:N]
                V = np.zeros_like(r_interior)  # No additional potential for Bessel
                a = np.zeros(N-1, dtype=float)
                b = np.zeros(N, dtype=float)
                c = np.zeros(N-1, dtype=float)
                b[0] = -1.0/(dr**2)
                c[0] = 1.0/(dr**2)
                for i in range(1, N):
                    a[i-1] = -1.0/(2*dr**2) + 1.0/(4*r_vals[i]*dr)
                    b[i] = 1.0/(dr**2)
                    if i < N:
                        c[i-1] = -1.0/(2*dr**2) - 1.0/(4*r_vals[i]*dr)
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
            elif wavefunction_type == "sech":
                # New spatial evolution branch for the "sech" wavefunction.
                # We'll assume a free evolution (V = 0) for simplicity.
                dt_eff = dt * slow_factor
                psi_complex = psi_complex.reshape((num_r, num_theta*num_phi))
                dr = r_vals[1] - r_vals[0]
                N = num_r - 1
                r_interior = r_vals[:N]
                V = np.zeros_like(r_interior)  # No potential term, for free spatial propagation.
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
                psi_updated[N, :] = 0.0  # enforce Dirichlet boundary condition
                psi_complex = psi_updated
                psi_flat = psi_complex.ravel()
            elif wavefunction_type == "hermite":
                dt_eff = dt * slow_factor
                psi_complex = psi_complex.reshape((num_r, num_theta*num_phi))
                dr = r_vals[1] - r_vals[0]
                N = num_r - 1
                # For the Hermite–Gaussian case, one simple option is to use a similar finite-difference
                # scheme as for the oscillator. Here, we use a harmonic oscillator potential as a proxy:
                omega_eff = 1.0  # adjust as needed based on hg_width or desired dynamics
                r_interior = r_vals[:N]
                V = 0.5 * omega_eff**2 * (r_interior**2)
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
                psi_updated[N, :] = 0.0  # enforce boundary conditions
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