# This script was developed using the EMAlign plugin as a template
import os
from chimerax.core import tools
from chimerax.core.commands import run
from chimerax.core.tools import ToolInstance
from chimerax.ui import MainToolWindow
from chimerax.ui.widgets import vertical_layout, button_row, ModelMenuButton, CollapsiblePanel, radio_buttons, EntriesRow
from Qt.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QDoubleSpinBox, QSpinBox, QVBoxLayout
from chimerax.map import Volume
from chimerax.map_data import ArrayGridData
from chimerax.map import volume_from_grid_data
import torch 
from .surfer import predict

# read help_info.json 
import json
help_info_path = os.path.join(os.path.dirname(__file__), "data", "help_info.json")
with open(help_info_path, "r") as f:
    help_info = json.load(f)

class SegmentMapTool(ToolInstance):
    def __init__(self, session, tool_name):
        super().__init__(session, tool_name)
        self.display_name = "LocScale-SURFER"
        self.tool_window = MainToolWindow(self)

        parent = self.tool_window.ui_area
        layout = vertical_layout(parent, margins=(5,0,0,0))
        self.log = session.logger

        # Create the pipeline panels for the two steps.
        self.pipeline_panel = self._create_pipeline_panel(parent)
        layout.addWidget(self.pipeline_panel)

        # Additional buttons for Options and Help.
        buttons_frame = self._create_action_buttons(parent)
        layout.addWidget(buttons_frame)

        # Advanced options panel.
        options = self._create_option_gui(parent)
        layout.addWidget(options)

        # Status label.
        self._status_label = QLabel(parent)
        layout.addWidget(self._status_label)

        self.tool_window.manage(placement="side")

        # Initially disable Step 2 until segmentation is complete.
        self._step2_frame.setEnabled(False)
        # The segmentation volume menu is initially disabled.
        self._segmentation_menu.setEnabled(False)
        self.segmented_map = None  # will hold the segmentation array
        self._segmentation_volume = None  # will hold the segmentation volume

    def _create_pipeline_panel(self, parent):
        panel = QFrame(parent)
        main_layout = vertical_layout(panel, margins=(0,0,0,0))

        # ----- Step 1: Segment Map -----
        step1_frame = QFrame(panel)
        step1_layout = vertical_layout(step1_frame, margins=(0,0,0,0))
        header1 = QLabel("<b>Step 1: Segment Map</b>", step1_frame)
        step1_layout.addWidget(header1)

        hframe1 = QFrame(step1_frame)
        hlayout1 = QHBoxLayout(hframe1)
        hlayout1.setContentsMargins(0,0,0,0)
        hlayout1.setSpacing(10)

        # Input map selection.
        input_label = QLabel("Input unsharpened map:", hframe1)
        input_map_help_text = help_info["input_map_help"]
        input_label.setToolTip(input_map_help_text)
        hlayout1.addWidget(input_label)
        self._query_map_menu = ModelMenuButton(self.session, class_filter=Volume)
#        vertical_list = self.session.models.list(type=Volume)
#        if vertical_list:
#            self._query_map_menu.value = vertical_list[0]
        self._query_map_menu.value_changed.connect(self._object_chosen)
        hlayout1.addWidget(self._query_map_menu)

        # Mask map selection (optional). Set autoselect="none" so that no default model is chosen.
        mask_label = QLabel("Mask (optional):", hframe1)
        mask_map_help_text = help_info["mask_map_help"]
        mask_label.setToolTip(mask_map_help_text)
        hlayout1.addWidget(mask_label)
        self._mask_map_menu = ModelMenuButton(self.session, class_filter=Volume,
                                               no_value_button_text="No model chosen",
                                               no_value_menu_text="None",
                                               autoselect="none")
        self._mask_map_menu.value_changed.connect(self._object_chosen)
        hlayout1.addWidget(self._mask_map_menu)

        hlayout1.addStretch(1)
        step1_layout.addWidget(hframe1)

        # "Segment" button in Step 1.
        hframe_seg = QFrame(step1_frame)
        seg_layout = QHBoxLayout(hframe_seg)
        seg_layout.setContentsMargins(0,0,0,0)
        seg_layout.setSpacing(10)
        self._segment_button = QPushButton("Segment", hframe_seg)
        self._segment_button.clicked.connect(self._segment)
        seg_layout.addWidget(self._segment_button)
        seg_layout.addStretch(1)
        step1_layout.addWidget(hframe_seg)

        main_layout.addWidget(step1_frame)

        # ----- Step 2: Detergent Removal -----
        self._step2_frame = QFrame(panel)
        step2_layout = vertical_layout(self._step2_frame, margins=(0,0,0,0))
        header2 = QLabel("<b>Step 2: Detergent Removal</b>", self._step2_frame)
        step2_layout.addWidget(header2)

        # Target map selection.
        hframe2 = QFrame(self._step2_frame)
        hlayout2 = QHBoxLayout(hframe2)
        hlayout2.setContentsMargins(0,0,0,0)
        hlayout2.setSpacing(10)
        target_label = QLabel("Target map:", hframe2)
        target_map_help_text = help_info["target_map_help"]
        target_label.setToolTip(target_map_help_text)
        hlayout2.addWidget(target_label)
        self._target_map_menu = ModelMenuButton(self.session, class_filter=Volume,
                                                no_value_button_text="No model chosen",
                                                autoselect="none")
        #if vertical_list:
        #    self._target_map_menu.value = vertical_list[0]
        self._target_map_menu.value_changed.connect(self._object_chosen)
        hlayout2.addWidget(self._target_map_menu)
        hlayout2.addStretch(1)
        step2_layout.addWidget(hframe2)

        # New row: "Use segmented map:" selection.
        hframe_seg_sel = QFrame(self._step2_frame)
        hlayout_seg_sel = QHBoxLayout(hframe_seg_sel)
        hlayout_seg_sel.setContentsMargins(0,0,0,0)
        hlayout_seg_sel.setSpacing(10)
        seg_sel_label = QLabel("Use segmented map:", hframe_seg_sel)
        hlayout_seg_sel.addWidget(seg_sel_label)
        self._segmentation_menu = ModelMenuButton(self.session, class_filter=Volume,
                                                  no_value_button_text="No model chosen",
                                                  autoselect="none")
        hlayout_seg_sel.addWidget(self._segmentation_menu)
        hlayout_seg_sel.addStretch(1)
        step2_layout.addWidget(hframe_seg_sel)

        # Threshold input for removal using QDoubleSpinBox.
        hframe_thresh = QFrame(self._step2_frame)
        thresh_layout = QHBoxLayout(hframe_thresh)
        thresh_layout.setContentsMargins(0,0,0,0)
        thresh_layout.setSpacing(10)
        thresh_label = QLabel("Removal Threshold:", hframe_thresh)
        theshold_label_help_text = help_info["threshold_help"]
        thresh_label.setToolTip(theshold_label_help_text)
        thresh_layout.addWidget(thresh_label)
        self._removal_threshold = QDoubleSpinBox(hframe_thresh)
        self._removal_threshold.setRange(0.0, 1.0)
        self._removal_threshold.setSingleStep(0.01)
        self._removal_threshold.setValue(0.5)
        thresh_layout.addWidget(self._removal_threshold)
        thresh_layout.addStretch(1)
        step2_layout.addWidget(hframe_thresh)

        # "Remove" button in Step 2.
        hframe_remove = QFrame(self._step2_frame)
        remove_layout = QHBoxLayout(hframe_remove)
        remove_layout.setContentsMargins(0,0,0,0)
        remove_layout.setSpacing(10)
        self._remove_button = QPushButton("Remove", hframe_remove)
        self._remove_button.clicked.connect(self._remove)
        remove_layout.addWidget(self._remove_button)
        remove_layout.addStretch(1)
        step2_layout.addWidget(hframe_remove)

        main_layout.addWidget(self._step2_frame)
        return panel

    def _create_action_buttons(self, parent):
        # Additional buttons for Options and Help.
        frame, buttons = button_row(parent, [
            ('Options', self._show_or_hide_options),
            ('Help', self._show_or_hide_guide)
        ], spacing=10, button_list=True)
        return frame

    # def _create_option_gui(self, parent):
    #     self._options_panel = CollapsiblePanel(parent, title='Advanced Options')
    #     content_area = self._options_panel.content_area

    #     # Prediction options.
    #     prediction_header = QLabel("<b>Prediction Options</b>", content_area)
    #     content_area.layout().addWidget(prediction_header)

    #     self._options_panel.setVisible(False)


    def _create_option_gui(self, parent):

        self._options_panel = CollapsiblePanel(parent, title=None)
        opt_layout = self._options_panel.layout()

        row = 1
        # --- Prediction Options ---
        pred_header = QLabel("<b>Prediction Options</b>", self._options_panel)
        opt_layout.addWidget(pred_header, row, 0)
        
        row += 1
        # Batch size option.
        batch_label = QLabel("Batch size:", self._options_panel)
        batch_size_help_text = help_info["batch_size_help"]
        batch_label.setToolTip(batch_size_help_text)
        self._batch_size = QSpinBox(self._options_panel)
        self._batch_size.setRange(1, 1024)
        self._batch_size.setValue(8)
        opt_layout.addWidget(batch_label, row, 0)
        opt_layout.addWidget(self._batch_size, row, 1)

        row += 1
        # Step size option.
        step_label = QLabel("Step size:", self._options_panel)
        step_size_info_text = f"Between 2 and 48"
        step_size_info_label = QLabel(step_size_info_text, self._options_panel)
        self._step_size = QSpinBox(self._options_panel)
        self._step_size.setRange(2, 48)
        self._step_size.setValue(32)
        opt_layout.addWidget(step_label, row, 0)
        opt_layout.addWidget(self._step_size, row, 1)
        opt_layout.addWidget(step_size_info_label, row, 2)

        row += 1
        # GPU ID option.
        is_gpu_available = torch.cuda.is_available()
        num_gpus_available = torch.cuda.device_count()
        gpu_label = QLabel("GPU ID:", self._options_panel)
        gpu_info_text = f"({num_gpus_available} GPUs available)" if is_gpu_available else "(No GPUs detected)"
        gpu_info_label = QLabel(gpu_info_text, self._options_panel)
        self._gpu_id = QSpinBox(self._options_panel)
        self._gpu_id.setRange(0, num_gpus_available - 1)
        self._gpu_id.setValue(0)
        opt_layout.addWidget(gpu_label, row, 0)
        opt_layout.addWidget(self._gpu_id, row, 1)
        opt_layout.addWidget(gpu_info_label, row, 2)

        gpu_label.setEnabled(is_gpu_available)
        self._gpu_id.setEnabled(is_gpu_available)

        row += 1
        # --- Removal Options ---
        rem_header = QLabel("<b>Removal Options</b>", self._options_panel)
        opt_layout.addWidget(rem_header, row, 0)

        row += 1
        smooth_label = QLabel("Smoothening filter size:", self._options_panel)
        smooth_label_info_text = f"Between 1 and 20"
        smooth_label_help_text = help_info["smooth_filter_help"]
        smooth_label_info_label = QLabel(smooth_label_info_text, self._options_panel)
        smooth_label.setToolTip(smooth_label_help_text)
        self._smooth_size = QSpinBox(self._options_panel)
        self._smooth_size.setRange(1, 20)
        self._smooth_size.setValue(5)
        opt_layout.addWidget(smooth_label, row, 0)
        opt_layout.addWidget(self._smooth_size, row, 1)
        opt_layout.addWidget(smooth_label_info_label, row, 2)
        
        self._options_panel.setVisible(False)
        
        return self._options_panel

    def _show_or_hide_options(self):
        is_options_visible = self._options_panel.isVisible()
        new_visibility = not is_options_visible
        self._options_panel.setVisible(new_visibility)
        

    def _show_or_hide_guide(self):
        from chimerax.help_viewer import show_url
        show_url(self.session, "https://gitlab.tudelft.nl/aj-lab/surfer")

    def _segment(self):
        import os
        from scipy.ndimage import uniform_filter
        input_map_for_segmentation = self._input_map()
        mask_map = self._mask_map()  # optional

        model_state_path = os.path.join(os.path.dirname(__file__), "data", "SURFER_SCUNet.pt")
        assert os.path.exists(model_state_path), f"Model state file not found at {model_state_path}"

        if input_map_for_segmentation is None:
            self._status_label.setText("No input map selected")
            return

        # Allow mask_map to be None.
        if mask_map is None:
            self.log.info("No mask map provided, proceeding without mask.")
            mask_map_np = None
        else:
            mask_map_np = mask_map.data.full_matrix()

        input_map_np = input_map_for_segmentation.data.full_matrix()
        pixel_size = input_map_for_segmentation.data.step
        origin = input_map_for_segmentation.data.origin
        # Print the status 
        self._status_label.setText("Segmenting map...")
        # Call predict with the prediction options.
        self.segmented_map = predict(
            input_map_np, pixel_size[0], mask_map_np,
            model_state_path=model_state_path,
            batch_size=self._batch_size.value(),
            step_size=self._step_size.value(),
            gpu_ids=[self._gpu_id.value()]
        )
        self._status_label.setText("Segmentation complete.")

        # Create a segmentation volume from the segmented map.
        seg_grid_data = ArrayGridData(self.segmented_map, origin=origin, step=pixel_size)
        seg_grid_data.name = "Predicted detergent micelle"
        seg_vol = volume_from_grid_data(seg_grid_data, self.session)
        self._segmentation_volume = seg_vol

        # Update the "Use segmented map:" menu with the segmentation volume.
        self._segmentation_menu.value = seg_vol
        self._segmentation_menu.setEnabled(True)

        # Enable Step 2.
        self._step2_frame.setEnabled(True)

    def _remove(self):
        from scipy.ndimage import uniform_filter
        target_map = self._target_map()
        if target_map is None:
            self._status_label.setText("No target map selected")
            return

        # Use the segmentation volume selected by the user.
        seg_vol = self._segmentation_menu.value
        if seg_vol is None:
            self._status_label.setText("No segmented map selected for removal")
            return

        # Get the segmentation data from the selected segmentation volume.
        seg_data = seg_vol.data.full_matrix()

        # Use the QDoubleSpinBox value for threshold.
        threshold_value = self._removal_threshold.value()
        # Binarise the segmentation data using the user-specified threshold.
        seg_binarised = (seg_data > threshold_value).astype(float)
        # Use the user-defined smoothening filter size.
        smooth_seg = uniform_filter(seg_binarised, size=self._smooth_size.value())

        target_map_np = target_map.data.full_matrix()
        new_target_map = target_map_np * (1 - smooth_seg)

        pixel_size = target_map.data.step
        origin = target_map.data.origin

        # Create new volume for target map without detergent.
        target_grid_data = ArrayGridData(new_target_map, origin=origin, step=pixel_size)
        target_grid_data.name = target_map.name + "_without_detergent"
        new_target_map_volume = volume_from_grid_data(target_grid_data, self.session)
        self._status_label.setText("Micelle removal complete.")

        # Display new target map and old target map at the same contour level for comparison.
        target_map.set_transparency(25)
        surface_level_of_target_map = target_map.minimum_surface_level
        run(self.session, "vop hide")
        new_target_map_volume.set_parameters(surface_levels=[surface_level_of_target_map], image_colors=[(0,1,0)])
        new_target_map_volume.show()
        target_map.show()


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
        self._s_map = self._input_map()
        self._m_map = self._mask_map()
        self._check_disable_options()

    def _check_disable_options(self):
        vlist = self.session.models.list(type=Volume)
        if not vlist:
            # If no volumes are available, you might disable related controls.
            pass

