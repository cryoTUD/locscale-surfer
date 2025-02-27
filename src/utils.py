#### collection locscale/emmernet functions ####
def average_voxel_size(voxel_size_record):
    apix_x = voxel_size_record.x
    apix_y = voxel_size_record.y
    apix_z = voxel_size_record.z
    
    average_apix = (apix_x+apix_y+apix_z)/3
    
    return average_apix

def load_map(map_path, return_apix = True, verbose=False):
    import mrcfile
    emmap = mrcfile.open(map_path).data
    apix = average_voxel_size(mrcfile.open(map_path).voxel_size)
    
    if verbose:
        print("Loaded map from path: ", map_path)
        print("Voxel size: ", apix)
        print("Map shape: ", emmap.shape)
        
    if return_apix:
        return emmap, apix
    else:
        return emmap
    
def convert_to_tuple(input_variable, num_dims=3):
    '''
    Convert any variable, or iterable into a tuple. If a scalar is input then a tuple is generated with same variable
    based on number of dimensions mentioned in num_dims

    Parameters
    ----------
    input_variable : any
        scalar, or any iterable
    num_dims : int, optional
        Length of tuple. The default is 3.
        
    Returns
    -------
    output_tuple : tuple

    '''
    
    if hasattr(input_variable, '__iter__'):
        if len(input_variable) == num_dims:
            output_tuple = tuple(input_variable)
            return output_tuple
        else:
            print("Input variable dimension {} doesn't match expected output dimension {}".format(len(input_variable), num_dims))
    else:
        output_list = [input_variable for temporary_index in range(num_dims)]
        output_tuple = tuple(output_list)
        return output_tuple

def save_as_mrc(map_data,output_filename, apix=None,origin=None,verbose=False, header=None):
    '''
    Function to save a numpy array containing volume, as a MRC file with proper header

    Parameters
    ----------
    map_data : numpy.ndarray
        Volume data showing the intensities of the EM Map at different points

    apix : float or any iterable
        In case voxelsize in x,y,z are all equal you can also just pass one parameter. 
    output_filename : str
        Path to save the MRC file. Example: 'path/to/map.mrc'
    origin: float or any iterable, optional
        In case origin index in x,y,z are all equal you can also just pass one parameter. 

    Returns
    -------
    Saves MRC .

    '''
    import numpy as np
    import mrcfile

    with mrcfile.new(output_filename,overwrite=True) as mrc:
        mrc.set_data(np.float32(map_data))
        
        if header is not None:
            mrc.set_extended_header(header)
        
        else:
            if apix is not None:
                #apix_list = [apix['x'], apix['y'], apix['z']]
                ## apix can be either a float or a list. If it's a single number, then the function convert_to_tuple will use it three times
                apix_tuple = convert_to_tuple(apix, num_dims=3)
                rec_array_apix = np.rec.array(apix_tuple, dtype=[('x','<f4'),('y','<f4'),('z','<f4')])
                mrc.voxel_size = rec_array_apix
            else:
                print("Please pass a voxelsize value either as a float or an iterable")
                return 0
            
            if origin is not None:    
                origin_tuple = convert_to_tuple(origin,num_dims=3)
            else:
                origin_tuple = convert_to_tuple(input_variable=0,num_dims=3)
            rec_array_origin = np.rec.array(origin_tuple, dtype=[('x','<f4'),('y','<f4'),('z','<f4')])
            mrc.header.origin = origin_tuple
            
        if verbose:
            print("Saving as MRC file format with following properties: ")
            print("File name: ", output_filename)
            print("Voxel size", mrc.voxel_size)
            print("Origin", mrc.header.origin)
            
        
    mrc.close()

    
def resample_map(emmap, emmap_size_new=None, apix=None, apix_new=None, order=1, assert_shape=None):
    '''
    Function to resample an emmap in real space using linear interpolation 

    Parameters
    ----------
    emmap : numpy.ndimage
        
    emmap_size_new : tuple 
        
    apix : float
        
    apix_new : float
        

    Returns
    -------
    resampled_emmap

    '''
    from scipy.ndimage import zoom
    if emmap_size_new is None:
        if apix is not None and apix_new is not None:
            resample_factor = apix/apix_new
        else:
            raise UserWarning("Provide either (1) current pixel size and new pixel size or (2) new emmap size")
    
    else:
        try:
            resample_factor = emmap_size_new[0] / emmap.shape[0]
        except:
            raise UserWarning("Please provide proper input: emmap_size_new must be a tuple")
    
    if assert_shape is not None:
        if isinstance(assert_shape, int):
            nx = assert_shape
        if isinstance(assert_shape, tuple):
            nx = assert_shape[0]
        if isinstance(assert_shape, list):
            nx = assert_shape[0]
        assertion_factor = nx / (emmap.shape[0] * resample_factor)
        resample_factor *= assertion_factor

    resampled_image = zoom(emmap, resample_factor, order=order, grid_mode=False)
    
    return resampled_image

def standardize_map(im):
    """ standardizes 3D density data

    Args:
        im (np.ndarray): 3D density data

    Returns:
        im (np.ndarray): standardized 3D density data
    """
    
    im = (im - im.mean()) / (10 * im.std())
    
    return im
    

def preprocess_emmap(emmap, apix, standardize):
    '''
    Function to preprocess the EM map path
    
    '''
    emmap_resampled = resample_map(emmap, apix=apix, apix_new=1.0, order=2)
    if standardize:
        emmap_standardized = standardize_map(emmap_resampled)
    else:
        emmap_standardized = emmap_resampled

    emmap_preprocessed = emmap_standardized
    return emmap_preprocessed

def extract_all_cube_centers(im_input, step_size, cube_size):
    '''
    Utility function to extract all cube centers from a 3D density map in a rolling window fashion
    
    '''
    length, width, height = im_input.shape

    # extract centers of all cubes in the 3D map based on the step size
    cubecenters = []
    for i in range(0, length, step_size):
        for j in range(0, width, step_size):
            for k in range(0, height, step_size):
                # i,j,k are corner of the cube 
                # we need to find the center of the cube
                center_k = k + cube_size//2
                center_j = j + cube_size//2
                center_i = i + cube_size//2

                # check if the center is within the map
                if center_k < length and center_j < width and center_i < height:
                    center_within_map = True
                else:
                    center_within_map = False
                
                # check if bounding box is within the map
                if k + cube_size < length and j + cube_size < width and i + cube_size < height:
                    bounding_box_within_map = True
                else:
                    bounding_box_within_map = False
                
                if center_within_map and bounding_box_within_map:
                    cubecenters.append((center_i, center_j, center_k))
                
                if center_within_map and not bounding_box_within_map:
                    # Check which dimension is out of bounds
                    if k + cube_size >= length:
                        diff  = k + cube_size - length
                        center_k = center_k - diff
                    if j + cube_size >= width:
                        diff  = j + cube_size - width
                        center_j = center_j - diff
                    if i + cube_size >= height:
                        diff  = i + cube_size - height
                        center_i = center_i - diff
                    cubecenters.append((center_i, center_j, center_k))
    
    return cubecenters

def filter_cubecenters_by_mask(cubecenters, mask, cube_size, signal_to_noise_cubes=None):
    '''
    Utility function to filter cube centers by a mask

    '''
    import random
    mask = (mask > 0.5).astype(int)
    print("Initial number of cubes: {}".format(len(cubecenters)))
    filtered_cubecenters = []
    signal_cubes_centers = []
    noise_cubes_centers = []
    for center in cubecenters:
        cube = extract_window(mask, center=center, size=cube_size)
        if cube.sum() > 10:
            signal_cubes_centers.append(center)
        else:
            noise_cubes_centers.append(center)

    num_signal_cubes = len(signal_cubes_centers)
    num_noise_cubes = len(noise_cubes_centers)
    if signal_to_noise_cubes is not None:
        required_noise_cubes = int(num_signal_cubes / signal_to_noise_cubes)
        if num_noise_cubes < required_noise_cubes:
            print("Not enough noise cubes. Using all noise cubes")
            sampled_noise_cubes = noise_cubes_centers
            
        else:
            print(f"Using {required_noise_cubes} noise cubes out of {num_noise_cubes} noise cubes randomly")
            sampled_noise_cubes = random.sample(noise_cubes_centers, required_noise_cubes)
        print(f"num_signal_cubes: {num_signal_cubes}")
        print(f"num_noise_cubes: {len(sampled_noise_cubes)}")
        
        filtered_cubecenters = signal_cubes_centers + sampled_noise_cubes
    else:
        filtered_cubecenters = signal_cubes_centers
        sampled_noise_cubes = None
    
    print(f"Number of cubes after filtering: {len(filtered_cubecenters)}")

    return filtered_cubecenters, signal_cubes_centers, sampled_noise_cubes

def extract_window(im, center, size):
    '''
    Extract a square window at a given location. 
    The center position of the window should be provided.

    Parameters
    ----------
    im : numpy.ndarray
        3D numpy array
    center : tuple, or list, or numpy.array (size=3)
        Position of the center of the window
    size : int, even
        Total window size (edge to edge) as an even number
        (In future could be modified to include different sized window 
        in different directions)
        

    Returns
    -------
    window : numpy.ndarray
        3D numpy array of shape (size x size x size)

    '''
    z,y,x = center
    window = im[z-size//2:z+size//2, y-size//2:y+size//2, x-size//2:x+size//2]
    return window

def cube_emmap(emmap, cubecenters, cube_size):
    '''adapted version of extract_cubes_from_cubecenters from emmernet for evaluation'''
    import numpy as np
    cubed_emmap = []
    for i, center in enumerate(cubecenters):
        cube = extract_window(emmap, center=center, size=cube_size)
        cube = np.expand_dims(cube, axis=3)

        cubed_emmap.append(cube)
    
    cubed_emmap = np.asarray(cubed_emmap)

    return cubed_emmap

def reassemble_map(pred_cubes, cubecenters, cube_size, out_shape):
    import numpy as np
    cube_mask = np.ones_like(pred_cubes)

    reassembly = np.zeros((out_shape))
    mask = np.zeros((out_shape))
    hcs = cube_size // 2

    pred_cubes = np.squeeze(pred_cubes)
    cube_mask  = np.squeeze(cube_mask)

    for i, center in enumerate(cubecenters):
        x,y,z = center
        cube = pred_cubes[i, ...]
        reassembly[x-hcs:x+hcs, y-hcs:y+hcs, z-hcs:z+hcs] += cube

        mcube = cube_mask[i, ...]
        mask[x-hcs:x+hcs, y-hcs:y+hcs, z-hcs:z+hcs] += mcube

    nonzero_indices = np.where(mask != 0)
    reassembly[nonzero_indices] /= mask[nonzero_indices]

    return reassembly
