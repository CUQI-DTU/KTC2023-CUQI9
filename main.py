import glob

# from ktc import model

DATA = sorted(glob.glob("data/TrainingData/data*.mat"))
TRUTH = sorted(glob.glob("data/GroundTruths/true*.mat"))

radius = 1
electrode_count = 32

# %% Create mesh

def create_disk_mesh(radius, polygons, cell_size):
    center = Point(0, 0)
    domain = Circle(center, radius, polygons)
    mesh = generate_mesh(domain, cell_size)
    return mesh


