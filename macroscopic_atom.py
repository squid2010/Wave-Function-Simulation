import glfw
from OpenGL.GL import *
import numpy as np
import ctypes
import time
from pyrr import Matrix44
import imgui
from imgui.integrations.glfw import GlfwRenderer

# Load atom parameters from external module
from atoms import atomic_species

# GLSL Shaders
vertex_src = '''
#version 330 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aUV;
out vec2 vUV;
void main(){ vUV = aUV; gl_Position = vec4(aPos,0,1); }'''

fragment_src = '''
#version 330 core
in vec2 vUV;
out vec4 FragColor;
uniform sampler2D tex;
uniform int mode;
uniform float zoom;
uniform float Lx,Ly;
uniform float Rx,Ry;
vec3 hsv2rgb(vec3 c){ vec3 p = abs(fract(c.xxx + vec3(0.0,2.0/3.0,1.0/3.0))*6.0 -3.0) -1.0; return c.z * mix(vec3(1.0), clamp(p,0.0,1.0), c.y); }
void main(){ vec2 z = (vUV - 0.5)/zoom + 0.5;
    float X = (z.x - 0.5)*Lx;
    float Y = (z.y - 0.5)*Ly;
    if((X*X)/(Rx*Rx) + (Y*Y)/(Ry*Ry) > 1.0) discard;
    float v = texture(tex, z).r;
    if(mode == 0) FragColor = vec4(v,v,v,1);
    else if(mode == 1) FragColor = vec4(v,v,v,1);
    else if(mode == 2){ vec3 col = hsv2rgb(vec3(v,1,1)); FragColor = vec4(col,1); }
    else FragColor = vec4(v,v,v,1);
}'''

def compile_shader(src, kind):
    sh = glCreateShader(kind)
    glShaderSource(sh, src)
    glCompileShader(sh)
    if not glGetShaderiv(sh, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(sh).decode())
    return sh

class AdaptiveGrid:
    """
    Quadtree-based adaptive mesh refinement for density mode
    """
    def __init__(self, base_Nx, base_Ny, max_depth=5, var_thresh=1e-3):
        self.base_Nx = base_Nx
        self.base_Ny = base_Ny
        self.max_depth = max_depth
        self.var_thresh = var_thresh
        self.vbo = glGenBuffers(1)
        self.ebo = glGenBuffers(1)
        self.vao = glGenVertexArrays(1)
        self.vcount = 0

    def refine(self, density):
        quads = [((0,0,self.base_Nx,self.base_Ny), 0)]
        mesh = []
        while quads:
            (x0,y0,dx,dy), depth = quads.pop()
            region = density[x0:x0+dx, y0:y0+dy]
            # adapt using variance threshold
            if depth < self.max_depth and np.var(region) > self.var_thresh:
                hx, hy = dx//2, dy//2
                quads += [((x0,    y0,   hx, hy), depth+1),
                          ((x0+hx, y0,   hx, hy), depth+1),
                          ((x0,    y0+hy,hx, hy), depth+1),
                          ((x0+hx, y0+hy,hx, hy), depth+1)]
            else:
                mesh.append((x0, y0, dx, dy))
        verts, idx = [], []
        for (x0, y0, dx, dy) in mesh:
            nx = np.array([x0, x0+dx]) / self.base_Nx
            ny = np.array([y0, y0+dy]) / self.base_Ny
            corners = [(nx[0],ny[0]), (nx[1],ny[0]), (nx[1],ny[1]), (nx[0],ny[1])]
            base = len(verts)//4
            for u,v in corners:
                verts += [-1 + 2*u, -1 + 2*v, u, v]
            idx += [base, base+1, base+2, base+2, base+3, base]
        self.vcount = len(idx)
        data = np.array(verts, dtype=np.float32)
        indices = np.array(idx, dtype=np.uint32)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,16,ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,16,ctypes.c_void_p(8))
        glBindVertexArray(0)

    def draw(self):
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, self.vcount, GL_UNSIGNED_INT, None)
        glBindVertexArray(self.vao)

class MacroscopicQuantumSimulator:
    def __init__(self, species='Rb87'):
        # UI and sim params
        self.species_list = list(atomic_species.keys())
        self.species_index = self.species_list.index(species)
        self.pot_types = ['Harmonic','Box','Lattice','DoubleWell']
        self.pot_index = 0
        self.N_atoms = 1e5
        self.lattice_depth=1e-30; self.lattice_period=5e-6
        self.box_hx=10e-6; self.box_hy=10e-6
        self.dw_sep=5e-6; self.dw_barrier=1e-30
        self.Nx=256; self.Ny=256
        self.dt=1e-6; self.slow=1.0
        self.use_diss=False; self.gamma=0.05
        self.mode=1; self.zoom=1.0
        self.start_time = time.time()
        
        # adaptive grid controls
        self.grid_var_thresh = 1e-3
        self.grid_max_depth = 5
        self.show_grid_lines = False
        self.start_time = time.time()
        
        self.init_physics(); self.init_opengl()

    def init_physics(self):
        params = atomic_species[self.species_list[self.species_index]]
        self.hbar = 1.0545718e-34
        self.m = params['mass']
        self.a_s = params['a_s']
        self.Lx = self.box_hx*2; self.Ly = self.box_hy*2
        self.dx = self.Lx/self.Nx; self.dy = self.Ly/self.Ny
        self.g = 4*np.pi*self.hbar**2*self.a_s/self.m
        self.omega_x=2*np.pi*50; self.omega_y=2*np.pi*50
        a_ho = np.sqrt(self.hbar/(self.m*np.sqrt(self.omega_x*self.omega_y)))
        self.mu = (self.hbar*np.sqrt(self.omega_x*self.omega_y)/2)*(15*self.N_atoms*self.a_s/a_ho)**(2/5)
        kx=2*np.pi*np.fft.fftfreq(self.Nx,self.dx)
        ky=2*np.pi*np.fft.fftfreq(self.Ny,self.dy)
        KX,KY = np.meshgrid(kx,ky,indexing='ij')
        self.T = self.hbar*(KX**2+KY**2)/(2*self.m)
        self.compute_potential()
        self.psi0 = self.init_wave()
        self.psi = self.psi0.copy()
        self.time = 0.0
        self.energy0 = self.total_energy(self.psi0)[3]
        self.metrics = {'norm_error':0.0,'rms_error':0.0,'energy_drift':0.0}

    def compute_potential(self):
        X = np.linspace(-self.Lx/2, self.Lx/2, self.Nx)
        Y = np.linspace(-self.Ly/2, self.Ly/2, self.Ny)
        self.X, self.Y = np.meshgrid(X, Y, indexing='ij')
        t = self.pot_index
        if t==0:
            self.V_ext = 0.5*self.m*(self.omega_x**2*self.X**2 + self.omega_y**2*self.Y**2)
        elif t==1:
            V0=1e-29
            mask=(np.abs(self.X)<=self.box_hx)&(np.abs(self.Y)<=self.box_hy)
            self.V_ext = V0*np.logical_not(mask)
        elif t==2:
            V0=self.lattice_depth; d=self.lattice_period
            self.V_ext = V0*(np.sin(2*np.pi*self.X/d)**2 + np.sin(2*np.pi*self.Y/d)**2)
        else:
            Vb=self.dw_barrier; sep=self.dw_sep
            base=0.5*self.m*(self.omega_x**2*self.Y**2)
            dw = Vb*(
                np.exp(-((self.X-sep/2)**2+self.Y**2)/(sep/4)**2)
                + np.exp(-((self.X+sep/2)**2+self.Y**2)/(sep/4)**2)
            )
            self.V_ext = base + dw
        self.Rx, self.Ry = self.Lx/2, self.Ly/2

    def init_wave(self):
        nTF = np.maximum((self.mu-self.V_ext)/self.g, 0)
        psi = np.sqrt(nTF).astype(np.complex128)
        norm = np.sqrt(np.sum(np.abs(psi)**2)*self.dx*self.dy)
        psi *= np.sqrt(self.N_atoms)/norm
        psi *= np.exp(1j*0.1*np.random.randn(*psi.shape))
        return psi

    def total_energy(self, psi):
        psi_k = np.fft.fft2(psi)
        kinetic = np.sum((self.hbar**2/(2*self.m))*np.abs(psi_k)**2 * (self.T/(self.hbar/(2*self.m))))
        kinetic *= self.dx*self.dy/(self.Nx*self.Ny)
        potential = np.sum(self.V_ext*np.abs(psi)**2) * self.dx*self.dy
        interaction = 0.5*self.g*np.sum(np.abs(psi)**4) * self.dx*self.dy
        return kinetic, potential, interaction, kinetic+potential+interaction

    def init_opengl(self):
        if not glfw.init(): raise RuntimeError()
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR,3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR,3)
        glfw.window_hint(glfw.OPENGL_PROFILE,glfw.OPENGL_CORE_PROFILE)
        self.win = glfw.create_window(1200,900,'MQ Sim',None,None)
        glfw.make_context_current(self.win)
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)
        imgui.create_context(); self.impl = GlfwRenderer(self.win)
        # compile shaders
        vs = compile_shader(vertex_src, GL_VERTEX_SHADER)
        fs = compile_shader(fragment_src, GL_FRAGMENT_SHADER)
        self.prog = glCreateProgram(); glAttachShader(self.prog, vs); glAttachShader(self.prog, fs); glLinkProgram(self.prog)
        # uniforms
        self.u_mode = glGetUniformLocation(self.prog,'mode')
        self.u_zoom = glGetUniformLocation(self.prog,'zoom')
        self.u_Lx = glGetUniformLocation(self.prog,'Lx'); self.u_Ly = glGetUniformLocation(self.prog,'Ly')
        self.u_Rx = glGetUniformLocation(self.prog,'Rx'); self.u_Ry = glGetUniformLocation(self.prog,'Ry')
        glUseProgram(self.prog); glUniform1i(glGetUniformLocation(self.prog,'tex'),0)
        # full-screen quad
        verts = np.array([-1,-1,0,0, 1,-1,1,0, 1,1,1,1, -1,1,0,1], dtype=np.float32)
        idx = np.array([0,1,2,2,3,0], dtype=np.uint32)
        self.quadVAO = glGenVertexArrays(1); self.quadVBO = glGenBuffers(1); self.quadEBO = glGenBuffers(1)
        glBindVertexArray(self.quadVAO)
        glBindBuffer(GL_ARRAY_BUFFER, self.quadVBO); glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.quadEBO); glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0); glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,16,ctypes.c_void_p(0))
        glEnableVertexAttribArray(1); glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,16,ctypes.c_void_p(8))
        glBindVertexArray(0)
        # texture
        self.tex = glGenTextures(1); glBindTexture(GL_TEXTURE_2D, self.tex)
        for p in [GL_TEXTURE_MIN_FILTER,GL_TEXTURE_MAG_FILTER,GL_TEXTURE_WRAP_S,GL_TEXTURE_WRAP_T]:
            glTexParameteri(GL_TEXTURE_2D,p,GL_LINEAR if 'FILTER' in str(p) else GL_CLAMP_TO_EDGE)
        glBindTexture(GL_TEXTURE_2D,0)
        # adaptive grid (after GL context ready)
        self.adaptive = AdaptiveGrid(self.Nx, self.Ny)

    def evolve(self):
        dt_eff = self.dt/(self.slow**3.75)
        psi_k = np.fft.fft2(self.psi); psi_k *= np.exp(-1j*self.T*dt_eff/self.hbar)
        psi = np.fft.ifft2(psi_k)
        H = self.V_ext + self.g*np.abs(psi)**2
        if self.use_diss: psi *= np.exp(-self.gamma*H*dt_eff/self.hbar)
        psi *= np.exp(-1j*H*dt_eff/self.hbar)
        psi_k = np.fft.fft2(psi); psi_k *= np.exp(-1j*self.T*dt_eff/self.hbar)
        self.psi = np.fft.ifft2(psi_k)
        self.time += dt_eff
        # metrics
        norm = np.sum(np.abs(self.psi)**2)*self.dx*self.dy
        self.metrics['norm_error'] = (norm - self.N_atoms)/self.N_atoms * 100
        r2 = np.sum((self.X**2+self.Y**2)*np.abs(self.psi)**2)*self.dx*self.dy / norm
        rms = np.sqrt(r2)
        tf_rms = np.sqrt((self.Rx**2 + self.Ry**2)/4)
        self.metrics['rms_error'] = (rms - tf_rms)/tf_rms * 100
        ek,ep,ei,et = self.total_energy(self.psi)
        self.metrics['energy_drift'] = (et - self.energy0)/abs(self.energy0) * 100

    def render(self):
        glClearColor(0.1,0.1,0.1,1); glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.prog)
        glUniform1i(self.u_mode, self.mode); glUniform1f(self.u_zoom, self.zoom)
        glUniform1f(self.u_Lx, self.Lx); glUniform1f(self.u_Ly, self.Ly)
        glUniform1f(self.u_Rx, self.Rx); glUniform1f(self.u_Ry, self.Ry)

        if self.mode == 1:
            # density mode with adaptive refinement
            density = np.abs(self.psi)**2
            normed = density / density.max()
            # refine mesh
            self.adaptive.refine(normed)
            # upload texture data
            glBindTexture(GL_TEXTURE_2D, self.tex)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, self.Nx, self.Ny, 0, GL_RED, GL_FLOAT, normed.astype(np.float32))
            glBindTexture(GL_TEXTURE_2D, 0)
            # draw refined mesh
            glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, self.tex)
            self.adaptive.draw()
        else:
            # full-screen quad for static/phase
            if self.mode == 0: vals = np.abs(self.psi0)**2
            elif self.mode == 2: vals = (np.angle(self.psi)/(2*np.pi)+0.5)%1.0
            else: vals = np.abs(self.psi)**2
            data = vals.astype(np.float32); data /= data.max()
            glBindTexture(GL_TEXTURE_2D, self.tex)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, self.Nx, self.Ny, 0, GL_RED, GL_FLOAT, data)
            glBindTexture(GL_TEXTURE_2D, 0)
            glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, self.tex)
            glBindVertexArray(self.quadVAO); glDrawElements(GL_TRIANGLES,6,GL_UNSIGNED_INT,None)
            glBindVertexArray(0)

    def run(self):
        while not glfw.window_should_close(self.win):
            glfw.poll_events(); self.impl.process_inputs(); imgui.new_frame()
            imgui.begin('Controls')
            changed,self.species_index=imgui.combo('Species',self.species_index,self.species_list)
            if changed: self.init_physics()
            _,self.N_atoms=imgui.input_float('N_atoms',self.N_atoms,format='%.0f')
            
            imgui.separator()
            
            imgui.text('Adaptive Grid:')
            changed, self.grid_var_thresh = imgui.slider_float('Var Threshold', self.grid_var_thresh, 1e-6, 1e-1, format='%.3e')
            changed2, self.grid_max_depth = imgui.slider_int('Max Depth', self.grid_max_depth, 1, 8)
            _, self.show_grid_lines = imgui.checkbox('Show Grid Lines', self.show_grid_lines)
                        
            imgui.separator()          
            
            changed,self.pot_index=imgui.combo('Potential',self.pot_index,self.pot_types)
            if changed: self.compute_potential(); self.psi0=self.init_wave(); self.psi=self.psi0.copy(); self.time=0.0
            if self.pot_index==1:
                _,self.box_hx=imgui.input_float('Box Hx',self.box_hx,format='%.2e'); _,self.box_hy=imgui.input_float('Box Hy',self.box_hy,format='%.2e')
            if self.pot_index==2:
                _,self.lattice_depth=imgui.input_float('Lattice Depth',self.lattice_depth,format='%.2e'); _,self.lattice_period=imgui.input_float('Lattice Period',self.lattice_period,format='%.2e')
            if self.pot_index==3:
                _,self.dw_sep=imgui.input_float('Well Separation',self.dw_sep,format='%.2e'); _,self.dw_barrier=imgui.input_float('Barrier Height',self.dw_barrier,format='%.2e')
            
            imgui.separator()
            
            _,self.dt=imgui.input_float('dt (s)',self.dt,format='%.1e')
            _,self.slow=imgui.slider_float('Slow Factor',self.slow,1.0,2.5e9)
            _,self.zoom=imgui.slider_float('Zoom',self.zoom,0.1,10.0)
            _,self.mode=imgui.combo('Mode',self.mode,['Static','Density','Phase'])
            _,self.use_diss=imgui.checkbox('Dissipation',self.use_diss)
            if self.use_diss: _,self.gamma=imgui.slider_float('Gamma',self.gamma,0.0,1.0)
            if imgui.button('Reinitialize'): self.init_physics()
            imgui.same_line()
            if imgui.button('Reset T'): self.time=0.0; self.psi=self.psi0.copy()
            imgui.end()
            imgui.begin('Metrics')
            kin,pot,inter,tot=self.total_energy(self.psi)
            imgui.text(f"Energy drift: {self.metrics['energy_drift']:.2f}%")
            imgui.text(f"Norm error: {self.metrics['norm_error']:.2f}%")
            imgui.text(f"RMS radius error: {self.metrics['rms_error']:.2f}%")
            fps=1.0/(time.time()-self.start_time+1e-9)
            imgui.text(f"FPS: {fps:.1f}")
            imgui.end()
            self.evolve(); self.render()
            imgui.render(); self.impl.render(imgui.get_draw_data()); glfw.swap_buffers(self.win)
        self.impl.shutdown(); glfw.terminate()

if __name__=='__main__':
    sim=MacroscopicQuantumSimulator()
    sim.run()
