import io
import sys
import math
import numpy as np
import qdarktheme
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import Qt, pyqtSignal

from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtCore import Qt, pyqtSignal
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import Qt, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from annotation_utils import display_cell_multi_marker
from annotation_utils import process_cell_polygon



class CellAnnotationWidget(QWidget):
  
    annotation_complete = pyqtSignal(dict)
    cell_changed = pyqtSignal(str)

    def __init__(self, sampled_cell_ids, cell_types = None):
        super().__init__()
        self.sampled_cell_ids = sampled_cell_ids
        self.current_index = 0
        self.annotations = {}
        self.cell_types = cell_types or ['a', 'b']
        
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        self.canvas = FigureCanvas(self.figure)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        
        # Set a fixed size for the widget
        self.setFixedSize(1000, 800)  # Adjust as needed
        
        # Create a container widget for the canvas with fixed size
        canvas_container = QWidget()
        canvas_container.setFixedSize(900, 600)  # Adjust as needed
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        
        self.canvas.setFixedSize(900, 600)  # Match container size
        canvas_layout.addWidget(self.canvas)
        
        layout.addWidget(canvas_container, alignment = Qt.AlignCenter)
        
        self.cell_id_label = QLabel(self)
        self.cell_id_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.cell_id_label)
        
        button_layout = QHBoxLayout()
        
        for cell_type in self.cell_types:
            btn = QPushButton(cell_type)
            btn.clicked.connect(lambda _, ct = cell_type: self.on_button_click(ct))
            button_layout.addWidget(btn)
            
        layout.addLayout(button_layout)
        
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("Previous")
        self.prev_btn.clicked.connect(self.show_previous)
        nav_layout.addWidget(self.prev_btn)
        
        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self.show_next)
        nav_layout.addWidget(self.next_btn)
        
        self.quit_btn = QPushButton("Quit")
        self.quit_btn.clicked.connect(self.quit_annotation)
        nav_layout.addWidget(self.quit_btn)
        
        layout.addLayout(nav_layout)
        self.show_cell_id()

    def on_button_click(self, cell_type):
        self.annotate(cell_type)

    def show_cell_id(self):
        cell_id = self.sampled_cell_ids[self.current_index]
        self.cell_id_label.setText(f"Cell ID: {cell_id}")
        self.cell_changed.emit(cell_id)

    def annotate(self, cell_type):
        cell_id = self.sampled_cell_ids[self.current_index]
        self.annotations[cell_id] = cell_type
        self.show_next()

    def show_next(self):
        if self.current_index < len(self.sampled_cell_ids) - 1:
            self.current_index += 1
            self.show_cell_id()
            self.cell_changed.emit(self.sampled_cell_ids[self.current_index])  # Emit signal here
    
    def show_previous(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_cell_id()
            self.cell_changed.emit(self.sampled_cell_ids[self.current_index])  # Emit signal here

    def quit_annotation(self):
        self.annotation_complete.emit(self.annotations)
        self.close()

##########

def annotation_loop(core_instance, show_markers, cell_types = None):
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(qdarktheme.load_stylesheet())
    widget = CellAnnotationWidget(core_instance.sampled_cells, cell_types = cell_types)

    def on_cell_changed(cell_id):
        widget.figure.clear()  # Clear the entire figure
        segment_data = core_instance.segments[cell_id]  # Get segment data for this cell
        # print(f"Segment data shape: {np.array(segment_data).shape}")
        centroid, coords = process_cell_polygon(segment_data)
        # print(f"Centroid: {centroid}")
        display_cell_multi_marker(core_instance.image,
                                  core_instance.marker_info,
                                  centroid,
                                  coords,
                                  show_markers,
                                  widget.figure)
        widget.canvas.draw()  # Redraw the entire canvas
        widget.canvas.flush_events()  # Ensure the GUI updates
        
    widget.cell_changed.connect(on_cell_changed)

    def on_complete(annotations):
        core_instance.annotations = annotations

    widget.annotation_complete.connect(on_complete)
    
    # Trigger initial plot
    initial_cell_id = core_instance.sampled_cells[0]
    on_cell_changed(initial_cell_id)
    
    widget.show()
    
    app.exec_()


