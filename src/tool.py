# This script was developed using the EMAlign plugin as a template
# read help_info.json
import json
import os

import torch
from chimerax.core.commands import run
from chimerax.core.tools import ToolInstance
from chimerax.map import Volume, volume_from_grid_data
from chimerax.map_data import ArrayGridData
from chimerax.ui import MainToolWindow
from chimerax.ui.widgets import (
    CollapsiblePanel,
    ModelMenuButton,
    button_row,
    vertical_layout,
)
from Qt.QtCore import Qt
from Qt.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
)

from .surfer import predict

help_info_path = os.path.join(os.path.dirname(__file__), "data", "help_info.json")
with open(help_info_path, "r") as f:
    help_info = json.load(f)


class SegmentMapTool(ToolInstance):
    def __init__(self, session, tool_name):
        super().__init__(session, tool_name)
        self.display_name = "LocScale-SURFER"
        self.tool_window = MainToolWindow(self)
        self.show_step2 = False
        self.segment_largest_component = (
            True  # Default to segmenting the largest component
        )
        self.segmented_map = None  # will hold the segmentation array
        self.binarised_segmented_map = (
            None  # will hold the binarised segmentation array
        )
        self._segmentation_volume = None  # will hold the segmentation volume
        self._binarised_segmented_volume = (
            None  # will hold the binarised segmentation volume
        )
        self._new_target_map_volume = (
            None  # will hold the new target map volume (without detergent)
        )
        self._filter_size = (
            5  # Default filter size for smoothing the binarised segmented map
        )
        self._need_to_filter_segmented_map = (
            True  # Flag to check if the filter has been applied
        )

        parent = self.tool_window.ui_area
        layout = vertical_layout(parent, margins=(5, 0, 0, 0))
        self.log = session.logger
        self.log.info("LocScale-SURFER tool initialized.")
        self.log.info("Segment and hide micelle densities in cryo-EM maps.")
        self.log.info(
            "Use an unsharpened map as input, and optionally a mask map to speed up computation"
        )

        # Additional buttons for Options and Help.
        buttons_frame = self._create_action_buttons(parent)
        layout.addWidget(buttons_frame)

        # Create the pipeline panels for the two steps.
        self.pipeline_panel = self._create_pipeline_panel(parent)
        layout.addWidget(self.pipeline_panel)

        # Advanced options panel.
        options = self._create_option_gui(parent)
        layout.addWidget(options)

        self.tool_window.manage(placement="side")

        # Initially disable Step 2 until segmentation is complete.
        self._control_step2_display(self.show_step2)
        # The segmentation volume menu is initially disabled.
        # self._segmentation_menu.setEnabled(False)

    def _control_step2_display(self, display):
        # Enable or disable the display of Step 2 based on the segmentation completion.
        self._step2_frame.setEnabled(display)

    def _create_pipeline_panel(self, parent):
        panel = QFrame(parent)
        main_layout = vertical_layout(panel, margins=(0, 0, 0, 0))

        # ----- Step 1: Segment Map -----
        step1_frame = QFrame(panel)
        step1_layout = vertical_layout(step1_frame, margins=(0, 0, 0, 0))
        header1 = QLabel("<b>Step 1: Segment Map</b>", step1_frame)
        step1_layout.addWidget(header1)

        # Input map selection
        hframe1 = QFrame(step1_frame)
        hlayout1 = QHBoxLayout(hframe1)
        hlayout1.setContentsMargins(0, 0, 0, 0)
        hlayout1.setSpacing(10)

        
        input_label = QLabel("Input halfmap 1:", hframe1)
        input_map_help_text = help_info["input_map_help"]
        input_label.setToolTip(input_map_help_text)
        hlayout1.addWidget(input_label)

        self._query_map_menu = ModelMenuButton(self.session, class_filter=Volume)
        #        vertical_list = self.session.models.list(type=Volume)
        #        if vertical_list:
        #            self._query_map_menu.value = vertical_list[0]
        self._query_map_menu.value_changed.connect(self._object_chosen)
        hlayout1.addWidget(self._query_map_menu)
        hlayout1.addStretch(1)
        step1_layout.addWidget(hframe1)

        # Input second half map selection 
        hframe2 = QFrame(step1_frame)
        hlayout2 = QHBoxLayout(hframe2)
        hlayout2.setContentsMargins(0, 0, 0, 0)
        hlayout2.setSpacing(10)

        input_label2 = QLabel("Input halfmap 2:", hframe2)
        input_map_help_text = help_info["input_map_help"]
        input_label2.setToolTip(input_map_help_text)
        hlayout2.addWidget(input_label2)

        self._query_map_menu2 = ModelMenuButton(self.session, class_filter=Volume)
        self._query_map_menu2.value_changed.connect(self._object_chosen)
        hlayout2.addWidget(self._query_map_menu2)
        hlayout2.addStretch(1)
        step1_layout.addWidget(hframe2)

        # Mask map selection (optional). Set autoselect="none" so that no default model is chosen.
        mask_frame = QFrame(step1_frame)
        mask_layout = QHBoxLayout(mask_frame)
        mask_layout.setContentsMargins(0, 0, 0, 0)
        mask_layout.setSpacing(10)

        mask_label = QLabel("Mask (optional):", mask_frame)
        mask_map_help_text = help_info["mask_map_help"]
        mask_label.setToolTip(mask_map_help_text)
        mask_layout.addWidget(mask_label)
        self._mask_map_menu = ModelMenuButton(
            self.session,
            class_filter=Volume,
            no_value_button_text="No model chosen",
            no_value_menu_text="None",
            autoselect="none",
        )
        self._mask_map_menu.value_changed.connect(self._object_chosen)
        mask_layout.addWidget(self._mask_map_menu)
        mask_layout.addStretch(1)
        step1_layout.addWidget(mask_frame)

        # "Segment" button in Step 1.
        hframe_seg = QFrame(step1_frame)
        seg_layout = QHBoxLayout(hframe_seg)
        seg_layout.setContentsMargins(0, 0, 0, 0)
        seg_layout.setSpacing(10)
        self._segment_button = QPushButton("Segment", hframe_seg)
        self._segment_button.clicked.connect(self._segment)
        seg_layout.addWidget(self._segment_button)
        seg_layout.addStretch(1)
        step1_layout.addWidget(hframe_seg)

        main_layout.addWidget(step1_frame)

        # ----- Step 2: Detergent Removal -----
        self._step2_frame = QFrame(panel)
        step2_layout = vertical_layout(self._step2_frame, margins=(0, 0, 0, 0))
        header2 = QLabel("<b>Step 2: Detergent Removal</b>", self._step2_frame)
        step2_layout.addWidget(header2)

        # Target map selection.
        hframe2 = QFrame(self._step2_frame)
        hlayout2 = QHBoxLayout(hframe2)
        hlayout2.setContentsMargins(0, 0, 0, 0)
        hlayout2.setSpacing(10)
        target_label = QLabel("Target map:", hframe2)
        target_map_help_text = help_info["target_map_help"]
        target_label.setToolTip(target_map_help_text)
        hlayout2.addWidget(target_label)
        self._target_map_menu = ModelMenuButton(
            self.session,
            class_filter=Volume,
            no_value_button_text="No model chosen",
            autoselect="none",
        )

        self._target_map_menu.value_changed.connect(self._object_chosen)
        hlayout2.addWidget(self._target_map_menu)
        hlayout2.addStretch(1)
        step2_layout.addWidget(hframe2)

        # New row: "Use segmented map:" selection.
        hframe_seg_sel = QFrame(self._step2_frame)
        hlayout_seg_sel = QHBoxLayout(hframe_seg_sel)
        hlayout_seg_sel.setContentsMargins(0, 0, 0, 0)
        hlayout_seg_sel.setSpacing(10)
        seg_sel_label = QLabel("Use segmented map:", hframe_seg_sel)
        hlayout_seg_sel.addWidget(seg_sel_label)
        self._segmentation_menu = ModelMenuButton(
            self.session,
            class_filter=Volume,
            no_value_button_text="No model chosen",
            autoselect="none",
        )
        hlayout_seg_sel.addWidget(self._segmentation_menu)
        hlayout_seg_sel.addStretch(1)
        step2_layout.addWidget(hframe_seg_sel)

        # Threshold input for removal using QDoubleSpinBox.
        hframe_thresh = QFrame(self._step2_frame)
        thresh_layout = QHBoxLayout(hframe_thresh)
        thresh_layout.setContentsMargins(0, 0, 0, 0)
        thresh_layout.setSpacing(10)
        thresh_label = QLabel("Removal Threshold:", hframe_thresh)
        theshold_label_help_text = help_info["threshold_help"]
        thresh_label.setToolTip(theshold_label_help_text)
        thresh_layout.addWidget(thresh_label)
        self._removal_threshold = QDoubleSpinBox(hframe_thresh)
        self._removal_threshold.setRange(0.0, 1.0)
        self._removal_threshold.setSingleStep(0.001)
        self._removal_threshold.setDecimals(3)
        self._removal_threshold.setValue(0.5)
        thresh_layout.addWidget(self._removal_threshold)
        thresh_layout.addStretch(1)
        step2_layout.addWidget(hframe_thresh)

        # Add a button to create a binarised segmented volume.
        hframe_show_binarised = QFrame(self._step2_frame)
        show_binarised_layout = QHBoxLayout(hframe_show_binarised)
        show_binarised_layout.setContentsMargins(0, 0, 0, 0)
        show_binarised_layout.setSpacing(10)
        self._show_binarised_button = QPushButton(
            "Show Binarised Micelle", hframe_show_binarised
        )
        self._show_binarised_button.clicked.connect(
            lambda: self._create_binarised_segmented_volume()
        )
        self._show_binarised_button.setToolTip("Show the binarised segmentation volume")
        show_binarised_layout.addWidget(self._show_binarised_button)
        show_binarised_layout.addStretch(1)
        step2_layout.addWidget(hframe_show_binarised)

        # "Remove" button in Step 2.
        hframe_remove = QFrame(self._step2_frame)
        remove_layout = QHBoxLayout(hframe_remove)
        remove_layout.setContentsMargins(0, 0, 0, 0)
        remove_layout.setSpacing(10)
        self._remove_button = QPushButton("Remove", hframe_remove)
        self._remove_button.clicked.connect(self._remove)
        remove_layout.addWidget(self._remove_button)
        remove_layout.addStretch(1)
        step2_layout.addWidget(hframe_remove)

        # "Toggle view" button in Step 2.
        hframe_toggle = QFrame(self._step2_frame)
        toggle_layout = QHBoxLayout(hframe_toggle)
        toggle_layout.setContentsMargins(0, 0, 0, 0)
        toggle_layout.setSpacing(10)

        hide_membrane_label = QLabel("Hide membrane", hframe_toggle)
        hide_membrane_label.setToolTip("Hide membrane")
        show_membrane_label = QLabel("Show membrane", hframe_toggle)
        show_membrane_label.setToolTip("Show membrane")
        self._toggle_slider = QSlider(Qt.Horizontal, hframe_toggle)
        self._toggle_slider.setRange(0, 1)
        self._toggle_slider.setTickPosition(QSlider.TicksBelow)
        self._toggle_slider.setTickInterval(1)
        self._toggle_slider.setToolTip("<- Hide membrane | Show membrane ->")
        self._toggle_slider.valueChanged.connect(lambda v: self._toggle_display(v == 1))

        toggle_layout.addWidget(hide_membrane_label)
        toggle_layout.addWidget(self._toggle_slider)
        toggle_layout.addWidget(show_membrane_label)
        toggle_layout.addStretch(1)
        step2_layout.addWidget(hframe_toggle)
        main_layout.addWidget(self._step2_frame)

        return panel

    def _create_action_buttons(self, parent):
        # Additional buttons for Options and Help.
        frame, buttons = button_row(
            parent,
            [
                ("Options", self._show_or_hide_options),
                ("Help", self._show_or_hide_guide),
            ],
            spacing=10,
            button_list=True,
        )
        return frame

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
        step_size_info_text = "Between 2 and 48"
        step_size_info_label = QLabel(step_size_info_text, self._options_panel)
        self._step_size = QSpinBox(self._options_panel)
        self._step_size.setRange(2, 48)
        self._step_size.setValue(32)
        opt_layout.addWidget(step_label, row, 0)
        opt_layout.addWidget(self._step_size, row, 1)
        opt_layout.addWidget(step_size_info_label, row, 2)

        row += 1
        # GPU ID option.
        is_gpu_available = self._is_gpu_available()
        num_gpus_available = self._get_gpu_count()
        gpu_label = QLabel("GPU ID:", self._options_panel)
        gpu_info_text = (
            f"({num_gpus_available} GPUs available)"
            if is_gpu_available
            else "(No GPUs detected)"
        )
        tooltip_text = "Select GPU ID for segmentation. If no GPUs are available, CPU will be used."
        gpu_info_label = QLabel(gpu_info_text, self._options_panel)
        self._gpu_id = QSpinBox(self._options_panel)
        self._gpu_id.setRange(0, num_gpus_available - 1)
        self._gpu_id.setValue(0)
        gpu_label.setToolTip(tooltip_text)
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
        smooth_label_info_text = "Between 1 and 20"
        smooth_label_help_text = help_info["smooth_filter_help"]
        smooth_label_info_label = QLabel(smooth_label_info_text, self._options_panel)
        smooth_label.setToolTip(smooth_label_help_text)
        self._smooth_size = QSpinBox(self._options_panel)
        self._smooth_size.setRange(1, 20)
        self._smooth_size.setValue(self._filter_size)
        # set _has_filter_size_changed to True when the value is changed
        self._smooth_size.valueChanged.connect(
            lambda value: setattr(self, "_need_to_filter_segmented_map", True)
        )

        opt_layout.addWidget(smooth_label, row, 0)
        opt_layout.addWidget(self._smooth_size, row, 1)
        opt_layout.addWidget(smooth_label_info_label, row, 2)

        # Add a checkbox to select whether to filter the segmented map based on the largest segment
        row += 1
        self._largest_segment_checkbox = QCheckBox(
            "Use largest segment", self._options_panel
        )
        self._largest_segment_checkbox.setChecked(self.segment_largest_component)
        self._largest_segment_checkbox.stateChanged.connect(
            lambda state: self.largest_segment_checkbox_state_changed(state)
        )
        opt_layout.addWidget(self._largest_segment_checkbox, row, 0)

        # Add checkbox to control display of step 2.
        row += 1
        self._show_step2_checkbox = QCheckBox(
            "Enable Detergent Removal", self._options_panel
        )
        self._show_step2_checkbox.setChecked(self.show_step2)
        self._show_step2_checkbox.stateChanged.connect(self._control_step2_display)
        opt_layout.addWidget(self._show_step2_checkbox, row, 0)

        self._options_panel.setVisible(False)

        return self._options_panel

    def largest_segment_checkbox_state_changed(self, state):
        # Update the segment_largest_component attribute based on the checkbox state.
        self.segment_largest_component = state == Qt.Checked
        self.log.info(
            f"Segment largest component set to: {self.segment_largest_component}"
        )
        self._need_to_filter_segmented_map = (
            True  # Set the flag to True to filter the segmented map again
        )

    def _is_gpu_available(self):
        # assume OS is Linux
        if torch.cuda.is_available():
            return True
        elif torch.backends.mps.is_available():
            return True
        else:
            return False

    def _get_gpu_count(self):
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        elif torch.backends.mps.is_available():
            return torch.mps.device_count()
        else:
            return 0

    def _show_or_hide_options(self):
        is_options_visible = self._options_panel.isVisible()
        new_visibility = not is_options_visible
        self._options_panel.setVisible(new_visibility)

    def _show_or_hide_guide(self):
        from chimerax.help_viewer import show_url

        show_url(self.session, "https://cryotud.github.io/locscale-surfer/")

    def _segment(self):
        import os

        input_map_for_segmentation = self._input_map()
        input_map_2_for_segmentation = self._input_map_2()
        mask_map = self._mask_map()  # optional

        model_state_path = os.path.join(
            os.path.dirname(__file__), "data", "SURFER_SCUNet.pt"
        )
        assert os.path.exists(model_state_path), (
            f"Model state file not found at {model_state_path}"
        )

        if input_map_for_segmentation is None:
            self.log.warning("Halfmap 1 is required for segmentation.")
            return
        if input_map_2_for_segmentation is None:
            self.log.warning("Halfmap 2 is required for segmentation.")
            return

        # Allow mask_map to be None.
        if mask_map is None:
            mask_map_np = None
        else:
            mask_map_np = mask_map.data.full_matrix()

        self.log.info("Segmenting map with the following inputs:")
        self.log.info(f"Input map: {input_map_for_segmentation.name}")

        if mask_map is not None:
            self.log.info(f"Mask map: {mask_map.name}")
        else:
            self.log.info("No mask map provided.")

        input_map_1_np = input_map_for_segmentation.data.full_matrix()
        input_map_2_np = input_map_2_for_segmentation.data.full_matrix()
        input_map_np = (input_map_1_np + input_map_2_np) / 2.0  # Average the two halfmaps
        pixel_size = input_map_for_segmentation.data.step
        origin = input_map_for_segmentation.data.origin
        # Print the status
        self.log.info("Starting segmentation...")
        # Call predict with the prediction options.
        self.segmented_map = predict(
            input_map_np,
            pixel_size[0],
            mask_map_np,
            model_state_path=model_state_path,
            batch_size=self._batch_size.value(),
            step_size=self._step_size.value(),
            gpu_ids=[self._gpu_id.value()],
        )
        self.log.info("Segmentation complete.")

        # Create a segmentation volume from the segmented map.
        seg_grid_data = ArrayGridData(
            self.segmented_map, origin=origin, step=pixel_size
        )
        seg_grid_data.name = "Predicted detergent micelle"
        seg_vol = volume_from_grid_data(seg_grid_data, self.session)
        self._segmentation_volume = seg_vol

        # Update the "Use segmented map:" menu with the segmentation volume.
        self._segmentation_menu.value = seg_vol
        self._segmentation_menu.setEnabled(True)

        # Enable Step 2.
        self.show_step2 = True
        self._control_step2_display(self.show_step2)

    def _remove(self):
        target_map = self._target_map()
        if target_map is None:
            self.log.warning("No target map selected for detergent removal.")
            return

        self._update_binarised_segmented_map()

        target_map_np = target_map.data.full_matrix()
        new_target_map = target_map_np * (1 - self.binarised_segmented_map)

        pixel_size = target_map.data.step
        origin = target_map.data.origin

        # Create new volume for target map without detergent.
        new_target_grid_data = ArrayGridData(
            new_target_map, origin=origin, step=pixel_size
        )
        new_target_grid_data.name = target_map.name + "_without_micelle"
        new_target_map_volume = volume_from_grid_data(
            new_target_grid_data, self.session
        )
        self._new_target_map_volume = new_target_map_volume
        self.log.info("Micelle removal complete.")

        # Display new target map and old target map at the same contour level for comparison.
        # target_map.set_transparency(25)
        surface_level_of_target_map = target_map.minimum_surface_level
        run(self.session, "vop hide")
        new_target_map_volume.set_parameters(
            surface_levels=[surface_level_of_target_map], image_colors=[(0, 1, 0)]
        )
        new_target_map_volume.show()
        # target_map.show()

    def _toggle_display(self, show_original):
        if show_original:
            # show the target map and hide the new_target_map volume
            target_map = self._target_map()
            new_target_map_volume = self._new_target_map()
            if new_target_map_volume is None:
                self.log.warning("No new target map volume available for display.")
                return
            if target_map is None:
                self.log.warning("No target map selected for display.")
                return
            target_map.display = True
            new_target_map_volume.display = False

        else:
            # hide the target map and show the new_target_map volume
            target_map = self._target_map()
            new_target_map_volume = self._new_target_map()
            if new_target_map_volume is None:
                self.log.warning("Remove detergent first")
                return
            if target_map is None:
                self.log.warning("Select target map")
                return
            surface_level_of_target_map = target_map.minimum_surface_level
            target_map.display = False
            new_target_map_volume.display = True
            new_target_map_volume.set_parameters(
                surface_levels=[surface_level_of_target_map], image_colors=[(0, 1, 0)]
            )

    def _input_map(self):
        m = self._query_map_menu.value
        return m if isinstance(m, Volume) else None

    def _input_map_2(self):
        m = self._query_map_menu2.value
        return m if isinstance(m, Volume) else None

    def _mask_map(self):
        m = self._mask_map_menu.value
        return m if isinstance(m, Volume) else None

    def _target_map(self):
        m = self._target_map_menu.value
        return m if isinstance(m, Volume) else None

    def _new_target_map(self):
        m = self._new_target_map_volume
        return m if isinstance(m, Volume) else None

    def _create_binarised_segmented_map(self, threshold):
        seg_vol = self._segmentation_menu.value
        if seg_vol is None:
            self.log.warning("No segmentation volume selected.")
            return

        # Get the segmentation data from the selected segmentation volume.
        seg_data = seg_vol.data.full_matrix()

        # Binarise the segmented map using the threshold.
        binarised_map = (seg_data > threshold).astype(float)
        self.binarised_segmented_map = binarised_map

    def _select_largest_component(self):
        from scipy.ndimage import label
        # if self.binarised_segmented_map is None:
        #     self.log.warning("Create binarised segmented map first")
        #     return None

        # Label the connected components in the binarised segmentation.
        labeled_seg, num_features = label(self.binarised_segmented_map)
        # Find the largest connected component.
        size_of_each_component = {
            i: (labeled_seg == i).sum() for i in range(1, num_features + 1)
        }
        largest_component = max(size_of_each_component, key=size_of_each_component.get)
        # Keep only the largest connected component.
        largest_component_mask = (labeled_seg == largest_component).astype(float)
        # Create a new binarised segmented map with only the largest component.
        self.binarised_segmented_map = largest_component_mask
        self.log.info("Largest component selected.")

    def _filter_binarised_segmented_map(self, filter_size):
        from scipy.ndimage import uniform_filter

        # Apply a uniform filter to smooth the binarised segmented map.
        if self._need_to_filter_segmented_map:
            smoothed_map = uniform_filter(
                self.binarised_segmented_map, size=filter_size
            )
            self.binarised_segmented_map = smoothed_map
            self._need_to_filter_segmented_map = False  # Reset the flag after filtering
            self.log.info(
                f"Binarised segmented map smoothed with filter size {filter_size}."
            )

    def _update_binarised_segmented_map(self):
        # Update the binarised segmented map with the current threshold value.
        threshold_value = self._removal_threshold.value()
        self._need_to_filter_segmented_map = True
        self._create_binarised_segmented_map(threshold_value)
        if self.segment_largest_component:
            # If the user wants to segment the largest component, we will do that.
            self._select_largest_component()

        # Smooth the binarised segmented map using the user-specified smooth size.
        self._filter_binarised_segmented_map(self._filter_size)

    def _create_binarised_segmented_volume(self):
        # Create binarised map
        pixel_size = self._input_map().data.step
        origin = self._input_map().data.origin

        self._update_binarised_segmented_map()

        self.log.info("Creating binarised segmented volume.")

        # Create a volume from the binarised segmented map.
        binarised_grid_data = ArrayGridData(
            self.binarised_segmented_map, origin=origin, step=pixel_size
        )
        binarised_grid_data.name = "Binarised Segmentation"
        binarised_volume = volume_from_grid_data(binarised_grid_data, self.session)
        self._binarised_segmented_volume = binarised_volume
        # set level to 0.5
        binarised_volume.set_parameters(surface_levels=[0.5])
        self.log.info("Binarised segmented volume created: " + binarised_volume.name)

    def _object_chosen(self):
        self._update_options()
        self.log.info(" ")

    def _update_options(self):
        self._s_map = self._input_map()
        self._m_map = self._mask_map()
        self._check_disable_options()

    def _check_disable_options(self):
        vlist = self.session.models.list(type=Volume)
        if not vlist:
            # If no volumes are available, you might disable related controls.
            pass
