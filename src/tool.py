## This file is created with the inspiration from EMalign4Chimerax tool 
# https://github.com/ShkolniskyLab/emalign4chimerax/blob/main/src/emalign_gui.py

from chimerax.core import tools
from chimerax.core.tools import ToolInstance
from chimerax.ui import MainToolWindow
from chimerax.ui.widgets import vertical_layout, button_row, ModelMenuButton, CollapsiblePanel, EntriesRow, radio_buttons
from Qt.QtWidgets import QFrame, QHBoxLayout, QLabel
from chimerax.map import Volume
from chimerax.map_data import ArrayGridData
from chimerax.map import volume_from_grid_data

from .surfer import predict


class SegmentMapTool(ToolInstance):
    def __init__(self, session, tool_name):
        super().__init__(session, tool_name)
        self.display_name = "LocScale-SURFER"
        self.tool_window = MainToolWindow(self)

        parent = self.tool_window.ui_area

        layout = vertical_layout(parent, margins=(5,0,0,0))

        self.log = session.logger

        # Make menus to choose input maps for segmentation and mask map for removing detergents
        maps_choice = self._create_surfer_map_menu(parent)
        layout.addWidget(maps_choice)

        # Make a button to start segmentation
        buttons_frame = self._create_action_buttons(parent)
        layout.addWidget(buttons_frame)

        # Make optional arguments for segmentation
        options = self._create_option_gui(parent)
        layout.addWidget(options)

        # Status line:
        self._status_label = sl = QLabel(parent)
        layout.addWidget(sl)

        self.tool_window.manage(placement="side")
    
    def _create_surfer_map_menu(self, parent):
        maps_frame = QFrame(parent)
        mlayout = QHBoxLayout(maps_frame)
        mlayout.setContentsMargins(0,0,0,0)
        mlayout.setSpacing(10)

        # Add a Menu button to select the input map

        segment_map = QLabel("Input unsharpened map:", maps_frame)
        mlayout.addWidget(segment_map)

        self._query_map_menu = qm = ModelMenuButton(self.session, class_filter=Volume)
        vertical_list = self.session.models.list(type = Volume)
        if vertical_list:
            qm.value = vertical_list[0]
        qm.value_changed.connect(self._object_chosen)
        mlayout.addWidget(qm)

        # Add a Menu button to select the mask map
        mask_map = QLabel("mask map:", maps_frame)
        mlayout.addWidget(mask_map)

        self._mask_map_menu = mm = ModelMenuButton(self.session, class_filter=Volume)
        mlayout.addWidget(mm)
        if vertical_list:
            mm.value = vertical_list[0]
        
        mm.value_changed.connect(self._object_chosen)

        # Add a Menu button to select the target map
        target_map = QLabel("Target map:", maps_frame)
        mlayout.addWidget(target_map)

        self._target_map_menu = tm = ModelMenuButton(self.session, class_filter=Volume)
        mlayout.addWidget(tm)
        if vertical_list:
            tm.value = vertical_list[0]

        tm.value_changed.connect(self._object_chosen)

        mlayout.addStretch(1)
    
        return maps_frame
    
    def _create_action_buttons(self, parent):
        # Buttons to segment, show options and show help
        f, buttons = button_row(parent, [('Segment', self._segment), 
                                         ('Options', self._show_or_hide_options), 
                                         ('Help', self._show_or_hide_guide)], spacing=10, button_list=True)
        
        return f

    def _create_guide(self, parent):
        self._guide_panel = g = CollapsiblePanel(parent, title=None)
        return g
    
    def _show_or_hide_guide(self):
        from chimerax.help_viewer import show_url
        show_url(self.session, "https://gitlab.tudelft.nl/aj-lab/surfer")
    
    def _create_option_gui(self, parent):
        # Advanced options, later to be added
        self._options_panel = op = CollapsiblePanel(parent, title='Options')
        self._options_panel.setVisible(False)
        return op
    
    def _show_or_hide_options(self):
        self._options_panel.toggle_panel_display()

    def _segment(self):
        import os 
        from scipy.ndimage import uniform_filter
        input_map_for_segmentation = self._input_map() 
        mask_map = self._mask_map()
        target_map = self._target_map()

        model_state_path = os.path.join(os.path.dirname(__file__), "dataDir", "model_20250208_103426_8.pt")
        assert os.path.exists(model_state_path), f"Model state file not found at {model_state_path}"

        if input_map_for_segmentation is None:
            self.status.setText("No input map selected")
            return
        if mask_map is None:
            self.status.setText("No mask map selected")
            return

        input_map_np = input_map_for_segmentation.data.full_matrix()
        pixel_size = input_map_for_segmentation.data.step 
        origin = input_map_for_segmentation.data.origin

        mask_map_np = mask_map.data.full_matrix()

        target_map_np = target_map.data.full_matrix()

        segmented_map = predict(input_map_np, pixel_size[0], mask_map_np, model_state_path=model_state_path)
        # Remove the detegernt micelle from the target map
        segmented_map_binarised = (segmented_map > 0.5).astype(float)
        smoothened_segmented_map = uniform_filter(segmented_map_binarised, size=5)

        new_target_map = target_map_np * (1 - smoothened_segmented_map)

        # Create a new map for segmented map
        segmented_map_grid_data = ArrayGridData(smoothened_segmented_map, origin=origin, step=pixel_size)
        segmented_map_grid_data.name = "Detergent micelle"
        segmented_volume = volume_from_grid_data(segmented_map_grid_data, self.session)

        # Create a new map for target map
        target_map_grid_data = ArrayGridData(new_target_map, origin=origin, step=pixel_size)
        target_map_grid_data.name = target_map.name + "_without_detergent"
        target_volume = volume_from_grid_data(target_map_grid_data, self.session)
        

    def _input_map(self):
        m = self._query_map_menu.value
        return m if isinstance(m, Volume) else None
    
    def _mask_map(self):
        m = self._mask_map_menu.value
        return m if isinstance(m, Volume) else None

    def _target_map(self):
        m = self._target_map_menu.value
        return m if isinstance(m, Volume) else None

    def _object_chosen(self):
        self._update_options()
        self._status_label.setText(" ")
    
    def _update_options(self):
        self._s_map = sm = self._input_map()
        self._m_map = mm = self._mask_map()

        self._check_disable_options() 
        if sm is None or mm is None:
            return
    
    def _check_disable_options(self):
        vlist = self.session.models.list(type = Volume)
        if not vlist: 
            for frame in self._all_frames:
                frame.setEnabled(False)
    

        


