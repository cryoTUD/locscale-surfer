from locscale.include.emmer.ndimage.map_utils import convert_pdb_to_mrc_position
import numpy as np
import gemmi
import random
from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist

def extract_dummy_residues_from_pdb(pdb_path):
    ''' obtain the coordinates of the planes given the pdb_path '''

    pdb_text = open(pdb_path, "r").readlines()
    dummy_atoms_lines = [line for line in pdb_text if (line.startswith("HETATM") and "DUM" in line)]
    atomic_coordinates_N = []
    atomic_coordinates_O = []
    for line in dummy_atoms_lines:
        x_coord = float(line[30:38])
        y_coord = float(line[38:46])
        z_coord = float(line[46:54])
        atom_name = line[12:16].strip()
        if atom_name == "N":
            atomic_coordinates_N.append([x_coord, y_coord, z_coord])
        elif atom_name == "O":
            atomic_coordinates_O.append([x_coord, y_coord, z_coord])
        
    return atomic_coordinates_N, atomic_coordinates_O

def find_z_planes_image(N_coord, O_coord, apix, nr_membranes, imsize):

    # N_coord, O_coord = extract_dummy_residues_from_pdb(pdb_path)

    mrc_pos_N = np.array(convert_pdb_to_mrc_position(N_coord, apix)) # convert to image coordinates
    freqN, _  = np.histogram(mrc_pos_N[:, 0], bins=imsize)           # obtain how often it occures
    max_indeces_N = np.argsort(freqN)[-nr_membranes:]                # get the largest indices
    z_coords_N = []

    for i in max_indeces_N:
        z_coords_N.append(mrc_pos_N[:, 0][i])

    mrc_pos_O = np.array(convert_pdb_to_mrc_position(O_coord, apix))
    freqO, _  = np.histogram(mrc_pos_O[:, 0], bins=imsize)
    max_indeces_O = np.argsort(freqO)[-nr_membranes:]
    z_coords_O = []


    for i in max_indeces_O:
        z_coords_O.append(mrc_pos_O[:, 0][i])

    return sorted(z_coords_N), sorted(z_coords_O)


def find_number_of_membranes(pdb_path):
    st = gemmi.read_structure(pdb_path)
    dummy_residues = []
    for model in st:
        for chain in model:
            for residue in chain:
                if residue.name == "DUM":
                    dummy_residues.append(residue)

    num_atoms_in_dummy_residues = [len(res) for res in dummy_residues]

    return max(num_atoms_in_dummy_residues)


def get_membrane(N_coord, O_coord, apix, frac_bias=0.40, imsize=230, axis="z"):

    x, y, z  = np.ogrid[0:imsize, 0:imsize, 0:imsize]

    if axis=="z":
        idx = 0
    elif axis=="y":
        idx = 1
    else:
        idx = 2
    
    lower_bound = min(convert_pdb_to_mrc_position(N_coord, apix)[0][idx], convert_pdb_to_mrc_position(O_coord, apix)[0][idx])
    upper_bound = max(convert_pdb_to_mrc_position(N_coord, apix)[0][idx], convert_pdb_to_mrc_position(O_coord, apix)[0][idx])

    mem_length = upper_bound - lower_bound
    bias = int(mem_length * frac_bias)

    # create membrane 
    membrane = ((1*x + 0*y + 0*z <= upper_bound +bias ) & (1*x + 0*y + 0*z >= lower_bound -bias)) *1.0

    return membrane

def parallel_plane_dist(planeA, planeB):
    # if plane 1 is:  ax + by + cz = - d1 
    # and plane 2 is: ax + by + cz = - d2
    # dist = |d2 – d1|/√(a^2 + b^2+ c^2)
    dist = abs(planeA[-1]-planeB[-1]) / np.sqrt(planeA[0]*planeA[0] + planeA[1]*planeA[1] + planeA[2]*planeA[2])
    return dist 

def find_all_points_on_plane(plane, imsize):
    a, b, c, d = plane
    e = 10**-14
    
    points = []
    for x in range(imsize):
        for y in range(imsize):
            for z in range(imsize):
                if ((a*x + b*y + c*z >= d-e) and (a*x + b*y + c*z <= d+e)):
                    points.append([x,y,z])

    return np.array(points)

def find_all_points_on_plane_vec(plane, imsize):
    abc = plane[:-1]
    d   = plane[-1]
    
    indices = np.arange(imsize)
    x, y, z = np.meshgrid(indices, indices, indices, indexing='ij')
    grid    = np.column_stack((x.ravel(), y.ravel(), z.ravel()))    # used chatgpt

    points = grid[grid.dot(abc) - d == 0]
    
    return points

def points_on_calc_plane(plane, coordinates, imsize):
    abc = plane[:-1]
    d   = plane[-1]

    points = coordinates[coordinates.dot(abc) - d == 0]

    return len(points)

def dist_3d(A, B):
    return np.linalg.norm(A-B)

def find_closest_dist(pointsA, pointsB):
    pointA = random.sample(pointsA, 1)

    dist = np.zeros(pointsB.shape[0])
    for i in range(len(pointsB)):
        dist[i] = dist_3d(pointA, pointsB[i,:])
    
    idx = np.argmin(dist)

    return dist[idx]

def find_average_dist(pointsA, pointsB):
    rep = 5
    avg_dist = 0
    for i in range(rep):
        avg_dist += find_closest_dist(pointsA, pointsB)
    
    avg_dist /= rep
    return avg_dist

def plane_dist(planeA, planeB):
    return np.linalg.norm(planeA - planeB)

def calculate_planes(N_coord, O_coord, apix):
    # N_set = [N_coord[0], N_coord[len(N_coord)//3], N_coord[-1]]
    # N_set = np.array(convert_pdb_toge_mrc_position(N_set, apix))

    # O_set = [O_coord[0], O_coord[len(O_coord)//3], O_coord[-1]]
    # O_set = np.array(convert_pdb_to_mrc_position(O_set, apix))

    N_set = random.sample(N_coord, 3)
    N_set = np.array(convert_pdb_to_mrc_position(N_set, apix))

    O_set = random.sample(O_coord, 3)
    O_set = np.array(convert_pdb_to_mrc_position(O_set, apix))

    A = np.flip(N_set[0,:])
    B = np.flip(N_set[1,:])
    C = np.flip(N_set[2,:])

    AB = B - A
    AC = C - A
    N_norm = np.cross(AB, AC) 
    N_norm = N_norm / np.linalg.norm(N_norm)
    N_norm = np.append(N_norm, A[0]*N_norm[0] + A[1]*N_norm[1] + A[2]*N_norm[2])

    K = np.flip(O_set[0,:])
    L = np.flip(O_set[1,:])
    M = np.flip(O_set[2,:])

    KL = L - K
    KM = M - K
    O_norm = np.cross(KL, KM) 
    O_norm = O_norm / np.linalg.norm(O_norm)
    O_norm = np.append(O_norm, K[0]*O_norm[0] + K[1]*O_norm[1] + K[2]*O_norm[2]) 

    return N_norm, O_norm

def get_membranes2(N_coord, O_coord, apix, frac_bias=0.40, imsize=230, nr_membranes=1):
    # create a grid
    x, y, z  = np.ogrid[0:imsize, 0:imsize, 0:imsize]

    if nr_membranes==1:
        '''takes rotation into account'''

        N_norm, O_norm = calculate_planes(N_coord, O_coord, apix)
        
        x, y, z  = np.ogrid[0:imsize, 0:imsize, 0:imsize]

        # create membrane 
        if N_norm[3] > O_norm[3]:
            membrane = ((N_norm[2]*x + N_norm[1]*y + N_norm[0]*z <= N_norm[3]) & (O_norm[2]*x + O_norm[1]*y + O_norm[0]*z >= O_norm[3])) *1.0
        else:
            membrane = ((N_norm[2]*x + N_norm[1]*y + N_norm[0]*z >= N_norm[3]) & (O_norm[2]*x + O_norm[1]*y + O_norm[0]*z <= O_norm[3])) *1.0
        
        # how to now do the bias? what is the membrane thickness?

    else:
        '''assumes that there is no rotation needed'''
        z_coords_N, z_coords_O = find_z_planes_image(N_coord, O_coord, apix, nr_membranes, imsize)

        membrane = (0*x + 0*y + 0*z != 0) *1.0 #shouldn't be satisfied anywhere, should be empty

        for i in range(nr_membranes):
            upper_bound = max(z_coords_N[i], z_coords_O[i])
            lower_bound = min(z_coords_N[i], z_coords_O[i])
            mem_length = upper_bound - lower_bound
            bias = int(mem_length * frac_bias)

            # add the membranes to an empty grid
            membrane += ((1*x + 0*y + 0*z <= upper_bound +bias ) & (1*x + 0*y + 0*z >= lower_bound -bias)) *1.0

    return membrane

### functions
def plane_opt(X,a,b,c,d):
    x,y = X
    return (-a*x + -b*y + d)/c

def calc_planes_scipy(coordinatesN, coordinatesO):
    x,y,z = coordinatesN.T
    popt, pcov = curve_fit(plane_opt, (x,y), z)
    planeN = popt/np.linalg.norm(popt[:-1])

    if (planeN[-1]<=0):
        planeN = -planeN

    x,y,z = coordinatesO.T
    popt, pcov = curve_fit(plane_opt, (x,y), z)
    planeO = popt/np.linalg.norm(popt[:-1])

    if (planeO[-1]<=0):
        planeO = -planeO

    membrane_thickness = get_membrane_thickness(coordinatesN, coordinatesO)

    return planeN, planeO, membrane_thickness

def calculate_planes(N_coord, O_coord, apix):
    ''' calculate plane by selecting three random points for each plane '''

    # select 3 random points N
    N_set = random.sample(N_coord, 3)
    N_set = np.array(convert_pdb_to_mrc_position(N_set, apix))

    # select 3 random points O
    O_set = random.sample(O_coord, 3)
    O_set = np.array(convert_pdb_to_mrc_position(O_set, apix))

    # calculate plane N
    A = N_set[0,:]
    B = N_set[1,:]
    C = N_set[2,:]

    AB = B - A
    AC = C - A
    N_norm = np.cross(AB, AC) 
    N_norm = N_norm / np.linalg.norm(N_norm)
    N_norm = np.append(N_norm, A[0]*N_norm[0] + A[1]*N_norm[1] + A[2]*N_norm[2])

    # calculate plane O
    K = O_set[0,:]
    L = O_set[1,:]
    M = O_set[2,:]

    KL = L - K
    KM = M - K
    O_norm = np.cross(KL, KM) 
    O_norm = O_norm / np.linalg.norm(O_norm)
    O_norm = np.append(O_norm, K[0]*O_norm[0] + K[1]*O_norm[1] + K[2]*O_norm[2]) 

    return N_norm, O_norm

def avg_dist_plane_points(plane, points):
    ''' distance of a point (x1,y1,z1) to a plane (ax + by + cz = d):
        d = abs( ax1 + by1 + cz1 - d) / sqrt (a^2 + b^2 + c^2) '''

    # plane = plane.reshape(4,1)
    abc = plane[:-1].copy()
    d   = plane[-1].copy()

    dist_vec = np.abs((np.matmul(points, abc) - d)) / np.sqrt(np.sum(np.square(abc)))
    return np.sum(dist_vec) / len(points)

def calculate_planes(N_coord, O_coord, apix):
    ''' calculate plane by selecting three random points for each plane '''

    # select 3 random points N
    N_set = random.sample(N_coord, 3)
    N_set = np.array(convert_pdb_to_mrc_position(N_set, apix))

    # select 3 random points O
    O_set = random.sample(O_coord, 3)
    O_set = np.array(convert_pdb_to_mrc_position(O_set, apix))

    # calculate plane N
    A = N_set[0,:]
    B = N_set[1,:]
    C = N_set[2,:]

    AB = B - A
    AC = C - A
    N_norm = np.cross(AB, AC) 
    N_norm = N_norm / np.linalg.norm(N_norm)
    N_norm = np.append(N_norm, A[0]*N_norm[0] + A[1]*N_norm[1] + A[2]*N_norm[2])

    # calculate plane O
    K = O_set[0,:]
    L = O_set[1,:]
    M = O_set[2,:]

    KL = L - K
    KM = M - K
    O_norm = np.cross(KL, KM) 
    O_norm = O_norm / np.linalg.norm(O_norm)
    O_norm = np.append(O_norm, K[0]*O_norm[0] + K[1]*O_norm[1] + K[2]*O_norm[2]) 

    return N_norm, O_norm

def get_membrane_thickness(pointsA, pointsB):
    dist = cdist(pointsA, pointsB, metric='euclidean') # used chatgpt to find function
    min_dist = np.min(dist, axis=1)
    avg_dist = min_dist.mean()
    return avg_dist

def avg_dist_plane_points(plane, points):
    ''' distance of a point (x1,y1,z1) to a plane (ax + by + cz = d):
        d = abs( ax1 + by1 + cz1 - d) / sqrt (a^2 + b^2 + c^2) '''

    # plane = plane.reshape(4,1)
    abc = plane[:-1].copy()
    d   = plane[-1].copy()

    dist_vec = np.abs((np.matmul(points, abc) - d)) / np.sqrt(np.sum(np.square(abc)))
    return np.sum(dist_vec) / len(points)

def find_best_plane(N_coord, O_coord, apix):
    early_stop = False

    # step 1: convert coordinates
    coordinatesN = np.array(convert_pdb_to_mrc_position(N_coord, apix))
    coordinatesO = np.array(convert_pdb_to_mrc_position(O_coord, apix))

    loops = 1000
    normsN = np.zeros((4,loops))
    normsO = np.zeros((4,loops))

    distN = np.zeros((loops, 1))
    distO = np.zeros((loops, 1))
        
    for i in range(loops):
        # step 2: calculate planes
        normsN[:,i], normsO[:,i] = calculate_planes(N_coord, O_coord, apix)
        
        # step 3: find the average distance to the plane
        distN[i] = avg_dist_plane_points(normsN[:,i], coordinatesN)
        distO[i] = avg_dist_plane_points(normsO[:,i], coordinatesO)

        if (distN[i] == 0 and distO[i] == 0): # probably only happens with perfect alignment
            early_stop = True
            idx = i
            break

    # step 4: find the best fit
    if early_stop:
        normN = normsN[:,idx]
        normO = normsO[:,idx]
    else: 
        idx = np.nanargmin(distN)
        normN = normsN[:,idx]
        idx = np.nanargmin(distO)
        normO = normsO[:,idx]

    # calculate the membrane thickness
    membrane_thickness = get_membrane_thickness(coordinatesN, coordinatesO)

    return normN, normO, membrane_thickness
    