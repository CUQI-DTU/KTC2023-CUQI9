import glob
import numpy as np
from mshr import *
from dolfin import *
# from ktc import model

DATA = sorted(glob.glob("data/TrainingData/data*.mat"))
TRUTH = sorted(glob.glob("data/GroundTruths/true*.mat"))

radius = 1
electrode_count = 32
electrode_width = np.pi / electrode_count
phase = np.pi / 2

# %% Create mesh

def create_disk_mesh(radius, electrode_count, polygons, cell_size):
    center = Point(0, 0)
    domain = Circle(center, radius, polygons)
    mesh = generate_mesh(domain, cell_size)

    class Electrode(SubDomain):
        def inside(self, x, on_boundary):
            r = np.linalg.norm(x)
            u, v = (np.cos(theta), np.sin(theta))
            rho = np.arccos(np.dot(x, [u, v]) / r)
            proj = np.maximum(2 * np.abs(rho), electrode_width)
            return on_boundary and np.isclose(proj, electrode_width)

    topology = mesh.topology()
    subdomains = MeshFunction("size_t", mesh, topology.dim() - 1)
    for i in range(electrode_count):
        theta = 2 * np.pi * i / electrode_count + phase
        electrode = Electrode()
        electrode.mark(subdomains, i + 1)

    return mesh, subdomains

mesh, subdomains = create_disk_mesh(radius, 32, 300, 50)

# %% Write subdomains to XDMF file
xdmf = XDMFFile("subdomains.xdmf")
xdmf.write(subdomains)
