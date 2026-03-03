import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

def project_map(emmap, projection_axis, projection_type="mean"):
    """
    Project the map along a given axis
    """
    if projection_type == "mean":
        fun = np.nanmean
    elif projection_type == "max":
        fun = np.nanmax
    elif projection_type == "min":
        fun = np.nanmin
    else:
        raise ValueError(f"Unknown projection type {projection_type}")

    if projection_axis == "x":
        
        return fun(emmap, axis=2)
    elif projection_axis == "y":
        return fun(emmap, axis=1)
    elif projection_axis == "z":
        return fun(emmap, axis=0)
    else:
        raise ValueError(f"Projection axis {projection_axis} is not valid. Choose from x, y, z")

def plot_projections(emmap, cmap="viridis", show_colorbar=False, projection_type="mean", return_figure=False, title=None):
    """
    Plot the projections of the map
    """
    fig, axes = plt.subplots(1, 3, figsize=(6, 18), dpi=300)
    
    projection_in_x = project_map(emmap, "x",projection_type)
    im_x=axes[0].imshow(projection_in_x, cmap=cmap)
    axes[0].set_title("X")
    # show axis colorbar
    if show_colorbar:
        divider = make_axes_locatable(axes[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im_x, cax=cax, orientation="vertical", cmap=cmap)
        
    projection_in_y = project_map(emmap, "y",projection_type)
    im_y=axes[1].imshow(projection_in_y, cmap=cmap)
    axes[1].set_title("Y")
    # show axis colorbar
    if show_colorbar:
        divider = make_axes_locatable(axes[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im_y, cax=cax, orientation="vertical", cmap=cmap)
        # hide y axis ticks
        axes[1].set_yticks([])
    
    axes[0].set_yticks([])

    projection_in_z = project_map(emmap, "z",projection_type)
    im_z=axes[2].imshow(projection_in_z, cmap=cmap)
    axes[2].set_title("Z")
    # show colorbar
    if show_colorbar:
        divider = make_axes_locatable(axes[2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im_z, cax=cax, orientation="vertical", cmap=cmap)
        # hide y axis ticks
        axes[2].set_yticks([])

    axes[0].set_yticks([])
    axes[1].set_yticks([])
    axes[2].set_yticks([])
    axes[0].set_xticks([])
    axes[1].set_xticks([])
    axes[2].set_xticks([])
    
    if title is not None:
        plt.suptitle(title)
    plt.tight_layout()
    if return_figure:
        return fig
    else:
        plt.show()

def plot_density_contour_map(vol):
    from skimage import measure
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    contour_low = 0.1
    contour_high = 0.99
    verts, faces, normals, values = measure.marching_cubes(vol, contour_low)
    # find contours of the highest density region
    verts_high, faces_high, normals_high, values_high = measure.marching_cubes(vol, contour_high)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    mesh_high = Poly3DCollection(verts_high[faces_high], alpha=0.1)
    
    face_color = [0.5, 0.5, 1]
    face_color_high = [1, 0, 0]

    mesh.set_facecolor(face_color)
    mesh_high.set_facecolor(face_color_high)

    ax.add_collection3d(mesh)
    ax.add_collection3d(mesh_high)
    # find the limiting X, Y and Z values of the mesh
    min_x, max_x = np.min(verts[:, 0]), np.max(verts[:, 0])
    min_y, max_y = np.min(verts[:, 1]), np.max(verts[:, 1])
    min_z, max_z = np.min(verts[:, 2]), np.max(verts[:, 2])
    # set the limits of the plot
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_zlim(min_z, max_z)
    
    plt.show()
    
def plot_density_contour_map_with_gradients(vol, gradient_x, gradient_y, gradient_z, stride=10, length=1, return_figure=False):
    from skimage import measure
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    norm = Normalize()

    all_contours = np.unique(vol.flatten())
    contour_low = np.percentile(all_contours, 5)
    contour_high = np.percentile(all_contours, 99)

    verts, faces, normals, values = measure.marching_cubes(vol, contour_low)
    # find contours of the highest density region
    verts_high, faces_high, normals_high, values_high = measure.marching_cubes(vol, contour_high)
    fig = plt.figure(figsize=(20, 20))
    # add two subplots
    ax = fig.add_subplot(121, projection='3d')
    
    
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    mesh_high = Poly3DCollection(verts_high[faces_high], alpha=0.1)
    
    face_color = [0.5, 0.5, 1]
    face_color_high = [1, 0, 0]

    mesh.set_facecolor(face_color)
    mesh_high.set_facecolor(face_color_high)

    ax.add_collection3d(mesh)
    ax.add_collection3d(mesh_high)

    # find the limiting X, Y and Z values of the mesh
    min_x, max_x = np.min(verts[:, 0]), np.max(verts[:, 0])
    min_y, max_y = np.min(verts[:, 1]), np.max(verts[:, 1])
    min_z, max_z = np.min(verts[:, 2]), np.max(verts[:, 2])
    # set the limits of the plot
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_zlim(min_z, max_z)
    ax.set_title(f"Contour map of density with low contour at {contour_low} and high contour at {contour_high}")

    # add the gradient vector field in the second subplot
    ax2 = fig.add_subplot(122, projection='3d')
    colormap = cm.inferno
    
    # Plot the gradient vector field
    Z,Y,X = np.mgrid[0:gradient_x.shape[0], 0:gradient_x.shape[1], 0:gradient_x.shape[2]]
    skip = (slice(None, None, stride), slice(None, None, stride), slice(None, None, stride))
    q = ax2.quiver(Z[skip], Y[skip], X[skip], gradient_z[skip], gradient_y[skip], gradient_x[skip], cmap = "seismic",  length=length, normalize=True, alpha = 1)
    # Set the arrow color based on the gradient magnitude 
    q.set_array(norm(vol.ravel()))

    # set the limits of the plot
    ax2.set_xlim(min_x, max_x)
    ax2.set_ylim(min_y, max_y)
    ax2.set_zlim(min_z, max_z)
    
    if return_figure:
        return fig
    else:
        plt.show()
        
def get_gradient_map(res, filename):
    dirname = f"data/maps/{res}gradient_label"
    with mrcfile.open(os.path.join(dirname, filename), permissive=True) as mrc:
        data = mrc.data
    return data

def plot_a_slice(gradient_map, index, orientation):
    "Gradient map is the 4 x 213 x 213 x 213 array, index is the index of the plane, orientation is x, y or z, \
     The result is a plot of a slice at the index when looking from the orientation direction"
    label_matrix = gradient_map[0]
    if orientation == 'x':
        plt.imshow(label_matrix[index, :, :])
    elif orientation == 'y':
        plt.imshow(label_matrix[:, index, :])
    elif orientation == 'z':
        plt.imshow(label_matrix[:, :, index])
    else:
        raise Exception("orientation must be either 'x', 'y' or 'z'")
        
def plot_rgb_gradient_slice(res, filename, index, orientation):
    "Plots a slice of the map with colors indicating the direction of the gradient in that 2D slice.\
     Also plots the semantic segmented slice for comparison. Index is the index to slice at and orientation is the \
     direction where you are looking from, e.g. 'x', 'y', or 'z'."
    
    # Get the map
    gradient_map = get_gradient_map(res, filename)
    
    # Separate the map into label and gradient matrices 
    label_mat = gradient_map[0]
    gz = gradient_map[1]
    gy = gradient_map[2]
    gx = gradient_map[3]
    
    # Select the gradient and label slice to be used, depending on the orientation and slicing index
    if orientation == 'x':
        g1_slice = gz[index, :, :]
        g2_slice = gy[index, :, :]
        label_mat_slice = label_mat[index, :, :]
    elif orientation == 'y':
        g1_slice = gx[:, index, :]
        g2_slice = gz[:, index, :]
        label_mat_slice = label_mat[:, index, :]
    elif orientation == 'z':
        g1_slice = gx[:, :, index]
        g2_slice = gy[:, :, index]
        label_mat_slice = label_mat[:, :, index]
        
    # Create an empty matrix for the gradient vector angles    
    angles = np.zeros([label_mat_slice.shape[0], label_mat_slice.shape[1]])
    
    # Fill the angles matrix
    for i in range(label_mat_slice.shape[0]):
        for j in range(label_mat_slice.shape[1]):
            g1, g2 = g1_slice[i, j], g2_slice[i, j]
            
            if g1 >= 0 and g2 >= 0:
                angle = np.pi + np.arctan(g2 / g1)
            elif g1 < 0 and g2 >= 0:
                angle = 3 * np.pi / 2 - np.arctan(g1 / g2)
            elif g1 < 0 and g2 < 0:
                angle = np.arctan(g2 / g1)
            elif g1 >= 0 and g2 < 0:
                angle = np.pi / 2 - np.arctan(g1 / g2)

            angles[i, j] = angle
    
    # Plot the results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))

    ax1.imshow(angles, cmap='hsv')
    ax2.imshow(label_mat_slice)

    plt.show()