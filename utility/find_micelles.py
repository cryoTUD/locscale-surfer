from locscale.include.emmer.ndimage.map_utils import convert_pdb_to_mrc_position
import numpy as np
import gemmi
from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture
from scipy import ndimage
import math

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

def plane_opt(X,a,b,c,d):
    '''func: ax+by+cz=d to optimize with scipy.optimize.curve_fit'''
    x,y = X
    return (-a*x + -b*y + d)/c

def get_membrane_thickness(pointsN, pointsO):
    '''calculate the average thickness'''
    dist = cdist(pointsN, pointsO, metric='euclidean') # used chatgpt to find function
    min_dist = np.min(dist, axis=1)
    avg_dist = min_dist.mean()
    return avg_dist

def calc_planes_scipy(coordinatesN, coordinatesO):
    '''find the best fit plane'''
    x,y,z = coordinatesN.T
    popt, pcov = curve_fit(plane_opt, (x,y), z)
    planeN = popt/np.linalg.norm(popt[:-1])

    if (planeN[-1]<=0):
        planeN = -planeN

    x,y,z = coordinatesO.T
    popt, pcov = curve_fit(plane_opt, (x,y), z)
    planeO = popt/np.linalg.norm(popt[:-1])

    if (planeO[-1]<0):
        planeO = -planeO

    membrane_thickness = get_membrane_thickness(coordinatesN, coordinatesO)

    return planeN, planeO, membrane_thickness

# def calculate_planes(N_coord, O_coord, apix):
def calculate_planes(coordinatesN, coordinatesO):
    ''' calculate plane by selecting three random points for each plane '''

    # select 3 random points N
    idx = np.random.randint(0, len(coordinatesN), size=3)
    N_set = coordinatesN[idx]
    # N_set = random.sample(N_coord, 3)
    # N_set = np.array(convert_pdb_to_mrc_position(N_set, apix))

    # select 3 random points O
    idx = np.random.randint(0, len(coordinatesO), size=3)
    O_set = coordinatesO[idx]
    # O_set = random.sample(O_coord, 3)
    # O_set = np.array(convert_pdb_to_mrc_position(O_set, apix))

    # calculate plane N
    A = N_set[0,:]
    B = N_set[1,:]
    C = N_set[2,:]

    AB = B - A
    AC = C - A
    N_norm = np.cross(AB, AC) 
    N_norm = N_norm / np.linalg.norm(N_norm)
    planeN = np.append(N_norm, A[0]*N_norm[0] + A[1]*N_norm[1] + A[2]*N_norm[2])

    # calculate plane O
    K = O_set[0,:]
    L = O_set[1,:]
    M = O_set[2,:]

    KL = L - K
    KM = M - K
    O_norm = np.cross(KL, KM) 
    O_norm = O_norm / np.linalg.norm(O_norm)
    planeO = np.append(O_norm, K[0]*O_norm[0] + K[1]*O_norm[1] + K[2]*O_norm[2]) 

    return planeN, planeO

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

# def find_best_plane(N_coord, O_coord, apix):
def find_best_plane(coordinatesN, coordinatesO):
    early_stop = False

    # # step 1: convert coordinates
    # coordinatesN = np.array(convert_pdb_to_mrc_position(N_coord, apix))
    # coordinatesO = np.array(convert_pdb_to_mrc_position(O_coord, apix))

    loops = 1000
    normsN = np.zeros((4,loops))
    normsO = np.zeros((4,loops))

    distN = np.zeros((loops, 1))
    distO = np.zeros((loops, 1))
        
    for i in range(loops):
        # step 2: calculate planes
        normsN[:,i], normsO[:,i] = calculate_planes(coordinatesN, coordinatesO)
        
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

    if (normN[-1]<=0):
        normN = -normN

    if (normO[-1]<0):
        normO = -normO

    # calculate the membrane thickness
    membrane_thickness = get_membrane_thickness(coordinatesN, coordinatesO)

    return normN, normO, membrane_thickness

def find_membrane_end(x,y,z, planeN, planeO, unmodelled_region, imsize):
    '''WARNING: pixels is not representative of the actual pixel count, due to the smoothening and filtering'''
    a,b,c,d = (planeN + planeO)/2

    smooth_region = ndimage.uniform_filter(unmodelled_region, size=5)

    sobel_x = ndimage.sobel(smooth_region, axis=0)
    sobel_y = ndimage.sobel(smooth_region, axis=1)
    sobel_z = ndimage.sobel(smooth_region, axis=2)

    # Combine the results
    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2 + sobel_z**2)

    laps = math.ceil(a*imsize + b*imsize + c*imsize)
    
    pixels, d, start = find_membrane_end_loop(x,y,z, planeN, planeO, laps, sobel_combined)

    if planeN[-1]>planeO[-1]:
        M1 = int(planeN[-1]-start) # first half of the membrane
        M2 = int(planeO[-1]-start) # second half of the membrane
    else:
        M2 = int(planeN[-1]-start)
        M1 = int(planeO[-1]-start)

    grad = np.gradient(pixels, d)
    lowerbound = min(np.argmax(grad[:M1])+start, min(planeN[-1], planeO[-1]))
    # lowerbound = np.argmax(grad[:M1])+start
    upperbound = max(np.argmin(grad[M2:])+start+M2, max(planeN[-1], planeO[-1]))
    # upperbound = np.argmin(grad[M2:])+start+M2
    return lowerbound, upperbound

def find_membrane_end_plot_purpose(x,y,z, planeN, planeO, unmodelled_region, imsize, flag='smoothening'):
    '''WARNING: pixels is not representative of the actual pixel count, due to the smoothening and filtering'''
    a,b,c,d = (planeN + planeO)/2

    if flag == 'smoothening':
        smooth_region = ndimage.uniform_filter(unmodelled_region, size=5)

        sobel_x = ndimage.sobel(smooth_region, axis=0)
        sobel_y = ndimage.sobel(smooth_region, axis=1)
        sobel_z = ndimage.sobel(smooth_region, axis=2)

        # Combine the results
        sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2 + sobel_z**2)
    else:
        sobel_combined = unmodelled_region

    laps = math.ceil(a*imsize + b*imsize + c*imsize)
    
    pixels, scan_range, start = find_membrane_end_loop(x,y,z, planeN, planeO, laps, sobel_combined)

    if planeN[-1]>planeO[-1]:
        M1 = int(planeN[-1]-start) # first half of the membrane
        M2 = int(planeO[-1]-start) # second half of the membrane
    else:
        M2 = int(planeN[-1]-start)
        M1 = int(planeO[-1]-start)

    grad = np.gradient(pixels, scan_range)
    lowerbound = min(np.argmax(grad[:M1])+start, min(planeN[-1], planeO[-1]))
    upperbound = max(np.argmin(grad[M2:])+start+M2, max(planeN[-1], planeO[-1]))

    return lowerbound, upperbound, grad, pixels, scan_range, planeN[-1], planeO[-1]

from numba import jit
@jit
def find_membrane_end_loop(x,y,z, planeN, planeO, laps, sobel_combined):
    a,b,c,d = (planeN + planeO)/2 # take the average plane since planes are not always completely parallel
    d = 0
    mem_length = abs(planeN[-1]-planeO[-1])

    '''obtain the starting and end points, to limit the scanning range'''
    start = int(max(0, min(planeN[-1] - mem_length, planeO[-1] - mem_length)))
    stop  = int(min(laps, max(planeN[-1] + mem_length, planeO[-1] + mem_length)))
    
    scan_range = np.arange(start, stop, 1)

    pixels = np.zeros(len(scan_range))

    ax_by_cz = a*x + b*y + c*z # precompute because of time

    '''find out the pixels in a slice'''
    for i, d in enumerate(scan_range):
        mem_slice = ((ax_by_cz >= d-0.5) & (ax_by_cz <= d+0.5)) *1.0
        pixels[i] = np.sum(mem_slice*sobel_combined)
    
    return pixels, scan_range, start

#chatgpt function
def map_clusters(nr_membranes, coordinatesN, coordinatesO, labelsN, labelsO): 
    # Initialize dictionary to store mappings from N clusters to O clusters
    cluster_mapping = {}
    
    # Iterate over each cluster in N coordinates
    for i in range(nr_membranes):
        # Get coordinates for the current N cluster
        sub_coordN = coordinatesN[labelsN == i]
        
        # Initialize variables to store minimum distance and corresponding cluster index
        min_distance = np.inf
        closest_cluster_index = None
        
        # Iterate over each cluster in O coordinates
        for j in range(nr_membranes):
            # Get coordinates for the current O cluster
            sub_coordO = coordinatesO[labelsO == j]
            
            # Calculate pairwise distances between points in N and O clusters
            distances = cdist(sub_coordN, sub_coordO)
            
            # Find the minimum distance
            min_dist_in_cluster = np.min(distances)
            
            # Update minimum distance and corresponding cluster index if necessary
            if min_dist_in_cluster < min_distance:
                min_distance = min_dist_in_cluster
                closest_cluster_index = j
        
        # Store the mapping between N cluster index and closest O cluster index
        cluster_mapping[i] = closest_cluster_index
    
    return cluster_mapping


import time

def get_membrane(N_coord, O_coord, unmodelled_region, apix, nr_membranes, imsize):
    '''for one or more membranes, obtain the regions where the membrane lies
       >> when multiple membrane regions are present, use kmeans clustering to sort the coordinates accordingly'''
    coordinatesN = np.array(convert_pdb_to_mrc_position(N_coord, apix))
    coordinatesO = np.array(convert_pdb_to_mrc_position(O_coord, apix))

    x, y, z  = np.ogrid[0:imsize, 0:imsize, 0:imsize]

    if nr_membranes==1:
        # calculate the planes and membrane thickness
        planeN, planeO, _ = find_best_plane(coordinatesN, coordinatesO)
        
        a,b,c,d = (planeN+planeO)/2
        # k,l,m,n = planeO
        tac = time.process_time()
        lowerbound, upperbound = find_membrane_end(x,y,z, planeN, planeO, unmodelled_region, imsize) 
        print(f'scan time: {time.process_time() - tac}')
        # use logical indexing to obtain the membrane region
        membrane = ((a*x + b*y + c*z <= upperbound) & (a*x + b*y + c*z >= lowerbound)) *1.0
        # if (d > n):
        #     membrane = ((a*x + b*y + c*z <= upperbound) & (k*x + l*y + m*z >= lowerbound)) *1.0
        # else:
        #     membrane = ((a*x + b*y + c*z >= lowerbound) & (k*x + l*y + m*z <= upperbound)) *1.0

    else:
        # cluster the coordinates with nr_membranes means (based on version sklearn 1.0.2)
        gmm = GaussianMixture(n_components=nr_membranes)
        gmm.fit(coordinatesN) 
        labelsN = gmm.predict(coordinatesN)

        gmm = GaussianMixture(n_components=nr_membranes)
        gmm.fit(coordinatesO) 
        labelsO = gmm.predict(coordinatesO)

        # create empty membrane to add regions to
        membrane = (0*x + 0*y + 0*z != 0) * 1.0

        map_dict = map_clusters(nr_membranes, coordinatesN, coordinatesO, labelsN, labelsO) #get the right combination of subcoords

        for i in range(nr_membranes):
            sub_coordN = coordinatesN[labelsN==i]
            j = map_dict[i]
            sub_coordO = coordinatesO[labelsO==j]

            # calculate the planes and membrane thickness
            tac = time.process_time()
            planeN, planeO, membrane_thickness = find_best_plane(sub_coordN, sub_coordO)
            print(f'scan time: {time.process_time() - tac}')

            a,b,c,d = (planeN+planeO)/2
            # k,l,m,n = planeO

            lowerbound, upperbound = find_membrane_end(x,y,z, planeN, planeO, unmodelled_region, imsize) 

            # use logical indexing to obtain the membrane region
            membrane += ((a*x + b*y + c*z <= upperbound) & (a*x + b*y + c*z >= lowerbound)) *1.0
            # if (d > n):
            #     membrane += ((a*x + b*y + c*z <= upperbound) & (k*x + l*y + m*z >= lowerbound)) *1.0
            # else:
            #     membrane += ((a*x + b*y + c*z >= lowerbound) & (k*x + l*y + m*z <= upperbound)) *1.0

    return membrane