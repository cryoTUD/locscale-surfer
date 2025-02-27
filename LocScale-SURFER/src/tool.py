from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QLabel, QPushButton, QMessageBox
from chimerax.core.models import Model  # just for type reference

class SegmentMapTool(QWidget):
    def __init__(self, session):
        super().__init__()
        self.session = session
        self.setWindowTitle("Segment Map Tool")
        self.build_ui()
        
    def build_ui(self):
        layout = QVBoxLayout()

        # Row for map model number
        map_layout = QHBoxLayout()
        map_label = QLabel("Map Model Number:")
        self.map_line_edit = QLineEdit()
        map_layout.addWidget(map_label)
        map_layout.addWidget(self.map_line_edit)
        layout.addLayout(map_layout)

        # Row for mask model number (optional)
        mask_layout = QHBoxLayout()
        mask_label = QLabel("Mask Model Number (optional):")
        self.mask_line_edit = QLineEdit()
        mask_layout.addWidget(mask_label)
        mask_layout.addWidget(self.mask_line_edit)
        layout.addLayout(mask_layout)

        # Button to run segmentation
        self.run_button = QPushButton("Segment Map")
        self.run_button.clicked.connect(self.run_segmentation)
        layout.addWidget(self.run_button)

        self.setLayout(layout)
    
    def get_model_by_number(self, model_text: str):
        """
        Given a model number as text (e.g. "#1" or "1"),
        return the corresponding volume model from the session.
        """
        model_text = model_text.strip()
        if model_text.startswith("#"):
            model_text = model_text[1:]
        try:
            model_id = int(model_text)
        except ValueError:
            return None
        
        # Look for a model with this id in the session's model list.
        for model in self.session.models.list():
            # Each model has an attribute "id" (an integer)
            if model.id == model_id:
                return model
        return None
    
    def run_segmentation(self):
        # Get the map model from the first field
        map_text = self.map_line_edit.text().strip()
        if not map_text:
            QMessageBox.warning(self, "Input Error", "Please enter a map model number.")
            return
        input_map = self.get_model_by_number(map_text)
        if input_map is None:
            QMessageBox.warning(self, "Input Error", f"Map model '{map_text}' not found.")
            return
        
        # Get the mask model (optional)
        mask_text = self.mask_line_edit.text().strip()
        input_mask = None
        if mask_text:
            input_mask = self.get_model_by_number(mask_text)
            if input_mask is None:
                QMessageBox.warning(self, "Input Error", f"Mask model '{mask_text}' not found.")
                return
        
        # Call the existing segment_map() function from cmd.py
        try:
            # Relative import from the same package
            from .cmd import segment_map
            segment_map(self.session, input_map, input_mask)
        except Exception as e:
            QMessageBox.critical(self, "Segmentation Error", f"Segmentation failed:\n{e}")
        else:
            QMessageBox.information(self, "Segmentation", "Segmentation completed successfully.")
