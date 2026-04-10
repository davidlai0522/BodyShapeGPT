import sys
import os
import threading
import traceback
import torch
import smplx
import numpy as np
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Prevent tokenizer deadlock

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, 
                             QVBoxLayout, QTextEdit, QPushButton, QLabel, QMessageBox, QFrame)
from PyQt5.QtCore import pyqtSignal, QObject, Qt
from PyQt5.QtGui import QFont
from pyvistaqt import QtInteractor
import pyvista as pv

# import logic from demo.py
from demo import load_model, run_model, parse_betas, DEFAULT_BASE_MODEL, DEFAULT_WEIGHTS_DIR

class WorkerSignals(QObject):
    model_loaded = pyqtSignal(object, object, object)  # processor, ft_model, smpl_model
    model_error = pyqtSignal(str)
    
    gen_done = pyqtSignal(object, object)  # betas, (vertices, faces)
    gen_error = pyqtSignal(str)

    status_update = pyqtSignal(str)


class BodyShapeGUI(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.processor = None
        self.ft_model = None
        self.smpl_model = None
        
        self.signals = WorkerSignals()
        self.signals.model_loaded.connect(self.on_model_loaded)
        self.signals.model_error.connect(self.show_error)
        self.signals.gen_done.connect(self.on_gen_done)
        self.signals.gen_error.connect(self.show_error)
        self.signals.status_update.connect(self.update_status)

        self.init_ui()
        
        # Start background load
        threading.Thread(target=self.bg_load_model, daemon=True).start()

    def init_ui(self):
        self.setWindowTitle("BodyShapeGPT - 3D Generator")
        self.resize(1000, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main Layout: HBox
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # -------- Left Panel (Controls) --------
        left_panel = QFrame()
        left_panel.setFixedWidth(350)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        title_font = QFont("Arial", 12, QFont.Bold)
        
        self.status_label = QLabel("Status: Waiting to load model...")
        self.status_label.setFont(title_font)
        self.status_label.setWordWrap(True)
        left_layout.addWidget(self.status_label)
        
        left_layout.addSpacing(20)
        
        input_label = QLabel("Input Description:")
        left_layout.addWidget(input_label)
        
        self.desc_input = QTextEdit()
        self.desc_input.setPlaceholderText("Describe the human body shape here...")
        self.desc_input.setText("Average height person with broad shoulders")
        self.desc_input.setMaximumHeight(150)
        left_layout.addWidget(self.desc_input)
        
        self.gen_btn = QPushButton("Generate && Visualize")
        self.gen_btn.setMinimumHeight(40)
        self.gen_btn.setStyleSheet("background-color: #2b78e4; color: white; font-weight: bold;")
        self.gen_btn.setEnabled(False)
        self.gen_btn.clicked.connect(self.on_generate_click)
        left_layout.addWidget(self.gen_btn)
        
        left_layout.addSpacing(20)
        
        self.params_label = QLabel("Shape params:\nN/A")
        self.params_label.setWordWrap(True)
        left_layout.addWidget(self.params_label)
        
        left_layout.addStretch()  # pushes everything up
        
        # -------- Right Panel (3D Viewer) --------
        right_panel = QFrame()
        right_panel.setStyleSheet("background-color: #1a1a1a;")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # PyVista 3D widget
        self.plotter = QtInteractor(right_panel)
        self.plotter.set_background("#1a1a1a")
        
        right_layout.addWidget(self.plotter.interactor)
        
        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)
        
    def update_status(self, text):
        self.status_label.setText(f"Status: {text}")

    def show_error(self, message):
        self.gen_btn.setEnabled(True)
        self.update_status("Error occurred.")
        QMessageBox.critical(self, "Error", message)

    # --- Background Tasks ---
    def bg_load_model(self):
        self.signals.status_update.emit("Loading language model (this may take a minute)...")
        try:
            processor, ft_model = load_model(self.args.model, self.args.weights, self.args.no_quantize)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            smpl = smplx.create(model_path=current_dir, model_type='smpl', gender=self.args.gender)
            self.signals.model_loaded.emit(processor, ft_model, smpl)
        except Exception as e:
            traceback.print_exc()
            self.signals.model_error.emit(str(e))

    def on_model_loaded(self, processor, ft_model, smpl):
        self.processor = processor
        self.ft_model = ft_model
        self.smpl_model = smpl
        self.update_status("Ready")
        self.gen_btn.setEnabled(True)

    def on_generate_click(self):
        desc = self.desc_input.toPlainText().strip()
        if not desc:
            QMessageBox.warning(self, "Invalid Input", "Please enter a description.")
            return
            
        self.gen_btn.setEnabled(False)
        self.update_status("Generating shape params...")
        threading.Thread(target=self.bg_generate, args=(desc,), daemon=True).start()

    def bg_generate(self, desc):
        try:
            raw = run_model(desc, self.processor, self.ft_model, self.args.max_new_tokens)
            betas = parse_betas(raw)
            
            # Prepare betas for SMPL
            padded_betas = list(betas)
            while len(padded_betas) < 10:
                padded_betas.append(0.0)
            padded_betas = padded_betas[:10]
            
            self.signals.status_update.emit("Running SMPL to create mesh...")
            custom_betas = torch.tensor([padded_betas], dtype=torch.float32)
            output = self.smpl_model(betas=custom_betas, return_verts=True)
            
            vertices = output.vertices.detach().cpu().numpy().squeeze() 
            faces = self.smpl_model.faces
            
            self.signals.gen_done.emit(padded_betas, (vertices, faces))
        except Exception as e:
            traceback.print_exc()
            self.signals.gen_error.emit(str(e))

    def on_gen_done(self, betas, mesh_data):
        self.update_status("Ready")
        self.gen_btn.setEnabled(True)
        self.params_label.setText(f"Shape params:\n{betas}")
        
        vertices, faces = mesh_data
        
        self.plotter.clear_actors()
        
        # PyVista requires faces array to be packed as [N, v1, v2, v3, ...]
        faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).ravel()
        
        mesh = pv.PolyData(vertices, faces_pv)
        
        # Render the mesh using VTK Physically Based Rendering
        self.plotter.add_mesh(
            mesh, 
            color="#ecdbcd",    # Neutral clay tone
            pbr=True,           
            metallic=0.1, 
            roughness=0.5,
            smooth_shading=True,
        )
        
        # Add cool screen-space ambient occlusion (SSAO) for deep shadows in the contours
        if hasattr(self.plotter, "enable_ssao"):
            self.plotter.enable_ssao(radius=0.5, bias=0.01)
            
        self.plotter.reset_camera()

def main():
    parser = argparse.ArgumentParser(description="BodyShapeGPT GUI (PyQt5)")
    parser.add_argument("--model", default=DEFAULT_BASE_MODEL, help=f"Base model HuggingFace ID")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS_DIR, help=f"Path to weights directory")
    parser.add_argument("--max-new-tokens", type=int, default=400, dest="max_new_tokens")
    parser.add_argument("--no-quantize", action="store_true", dest="no_quantize")
    parser.add_argument("--gender", choices=["male", "female", "neutral"], default="neutral")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    
    # Modern darkish fusion theme
    app.setStyle("Fusion")
    
    gui = BodyShapeGUI(args)
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
