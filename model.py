import numpy as np
from dolfin import MeshFunction, Point, SubDomain
from mshr import Circle, generate_mesh


def create_disk_mesh(radius, electrode_count, polygons=300, fineness=50):
    """
    Create a mesh representation of a disk and subdomains representing the
    electrodes.

    The electrodes are evenly spaced on the boundary of the disk and the
    width of each electrode is `pi / n` where `n` is the electrode count.

    Parameters
    ----------
    radius : float
        The radius of the disk.

    electrode_count : int
        The number of electrodes on the boundary of the disk.

    polygons : int, optional.

    fineness : float, optional.

    Returns
    -------
    mesh : dolfin.Mesh
        A mesh representation of the disk.

    subdomians : dolfin.MeshFunction
        The subdomains mark the electrodes on the boundary of the disk counting
        anticlockwise from 12-o'clock. The first electrode is marked with 0.
    """

    center = Point(0, 0)
    domain = Circle(center, radius, polygons)
    mesh = generate_mesh(domain, fineness)
    electrode_width = np.pi / electrode_count
    phase = np.pi / 2

    class Electrode(SubDomain):
        def __init__(self, theta, width):
            super().__init__()
            self.theta = theta
            self.width = width

        def inside(self, x, on_boundary):
            r = np.linalg.norm(x)
            u, v = (np.cos(self.theta), np.sin(self.theta))

            # Compute the normalised projection of x onto (u, v)
            proj = np.clip(np.dot(x, [u, v]) / r, -1, 1)
            rho = np.arccos(proj)

            # Project the angle to the edge of the electrode
            proj = np.maximum(2 * np.abs(rho), self.width)
            return on_boundary and np.isclose(proj, self.width)

    topology = mesh.topology()
    subdomains = MeshFunction("size_t", mesh, topology.dim() - 1)

    for i in range(electrode_count):
        theta = 2 * np.pi * i / electrode_count + phase
        electrode = Electrode(theta, electrode_width)
        electrode.mark(subdomains, i)

    return mesh, subdomains
