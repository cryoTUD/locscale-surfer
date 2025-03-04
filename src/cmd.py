"""
Delft University of Technology (TU Delft) hereby disclaims
all copyright interest in the program “LocScale-SURFER” written by
the Author(s).

Copyright (C) 2025 Arjen J. Jakobi (TU Delft), Alok Bhradwaj (TU Delft) and Lotte Verbeek (TU Delft)

"""
from chimerax.core.commands import CmdDesc      # Command description
import chimerax

# from chimerax.atomic import AtomsArg            # Collection of atoms argument
# from chimerax.core.commands import BoolArg      # Boolean argument
# from chimerax.core.commands import ColorArg     # Color argument
# from chimerax.core.commands import IntArg       # Integer argument
# from chimerax.core.commands import EmptyArg     # (see below)
# from chimerax.core.commands import Or, Bounded  # Argument modifiers
from chimerax.core.commands import ModelArg     # Model argument
from chimerax.map_data import ArrayGridData
from chimerax.map import volume_from_grid_data

# from chimerax.core.commands import run

# Import other python modules to segment_map function
# ==========================================================================
# Functions and descriptions for registering using ChimeraX bundle API
# ==========================================================================

def segment_map(session, inputMap=None, inputMask=None):
    ''' 
    Function to segment an input map containing a detergent micelle 
    '''
    assert inputMap is not None, "Please provide a map to segment"
    import os 
    # Import necessary packages 
    from .surfer import predict
    #model_state_path = "U:\ajlab\AB\parking_for_files\fromDB\segmentation_micelle_with_curateD_micelle_outputdata\20250208_103426_segmentation_scunet_curated_cz48_BCEloss_without_dilation\saved_models\model_20250208_103426_8.pt"
    print("__file__", __file__)
    model_state_path = os.path.join(os.path.dirname(__file__), "dataDir", "model_20250208_103426_8.pt")
    print("model_state_path", model_state_path)
    # check if the model_state_path exists
    if not os.path.exists(model_state_path):
        raise FileNotFoundError(f"Model state file not found at {model_state_path}")
    else:
        print(f"Model state file found at {model_state_path}")

    # Log the start of the segmentation
    session.logger.info("Segmenting map...")

    # Get the map data
    emmap = inputMap.data.full_matrix()
    apix = inputMap.data.step
    origin = inputMap.data.origin

    # If a mask is provided, get the mask data
    if inputMask is not None:
        mask = inputMask.data.full_matrix()
    else:
        mask = None

    output_array = predict(
                        emmap, apix[0], mask,\
                        batch_size = 8, cube_size = 48, step_size = 32, gpu_ids = [0], model_state_path=model_state_path
                    )

    # Create a new map
    new_map = ArrayGridData(output_array, origin=origin, step=apix)

    new_map.name = "Segmented map"
    print(f"New map name: {new_map.name}")
    new_volume = volume_from_grid_data(new_map, session)
        
    
surfer_desc = CmdDesc(optional=[("inputMap", ModelArg),("inputMask", ModelArg)])

def open_segmentation_tool(session):
    """
    Opens a GUI tool that allows the user to input model numbers
    for the map and mask and then run the segmentation.
    """
    from .tool import SegmentMapTool
    # Create an instance of the tool widget.
    tool_widget = SegmentMapTool(session)
    # Add the tool widget to the ChimeraX UI (here, in the side area).
    session.ui.tools.add_tool(tool_widget, area="side", floating=False)

# Command descriptor for opening the tool.
open_tool_desc = CmdDesc(optional=[])

def register_tool_command(logger):
    from chimerax.core.commands import register
    register("open segment tool", open_segmentation_tool, open_tool_desc, logger=logger)
                        