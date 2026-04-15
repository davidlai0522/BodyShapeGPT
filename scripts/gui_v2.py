"""
gui_v2.py — BodyShapeGPT Semantic Search GUI

The user types a natural-language body description.  Qwen-2.5-0.5B embeds
both the query and all 21 k dataset entries; the closest cosine-similarity
match is retrieved and its SMPL betas are rendered as a 3-D body mesh.

Embeddings are cached to disk so they are only computed once.
"""

import sys
import os
import json
import threading
import traceback
import argparse

import torch
import numpy as np
import smplx
import pyvista as pv

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer, AutoModel
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QTextEdit, QPushButton, QLabel, QMessageBox, QFrame, QProgressBar,
)
from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtGui import QFont
from pyvistaqt import QtInteractor

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR   = os.path.dirname(_SCRIPT_DIR)

DEFAULT_MODEL       = "Qwen/Qwen2.5-0.5B"
DEFAULT_DATASET     = os.path.join(_ROOT_DIR, "BodyShapeGPT_dataset.jsonl")
DEFAULT_EMBED_CACHE = os.path.join(_ROOT_DIR, "dataset_embeddings.npy")
DEFAULT_GENDER      = "neutral"

# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool last hidden states over non-padding tokens."""
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden_states * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def embed_texts(
    texts: list,
    tokenizer,
    model,
    device: str,
    batch_size: int = 64,
    max_length: int = 128,
    progress_cb=None,
) -> np.ndarray:
    """
    Return L2-normalised float32 embeddings, shape (N, hidden_size).

    progress_cb(done: int, total: int) is called after each batch.
    """
    results = []
    total = len(texts)
    for start in range(0, total, batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(device)
        with torch.no_grad():
            out = model(**enc)
        vecs = _mean_pool(out.last_hidden_state, enc["attention_mask"])
        vecs = torch.nn.functional.normalize(vecs.float(), dim=-1)
        results.append(vecs.cpu().numpy())
        if progress_cb:
            progress_cb(min(start + batch_size, total), total)
    return np.concatenate(results, axis=0)


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_dataset(path: str):
    descriptions, betas_list = [], []
    with open(path) as fh:
        for line in fh:
            entry = json.loads(line)
            descriptions.append(entry["description"])
            betas_list.append(json.loads(entry["shape_params"]))
    return descriptions, betas_list


# ---------------------------------------------------------------------------
# Qt worker signals
# ---------------------------------------------------------------------------

class WorkerSignals(QObject):
    progress = pyqtSignal(int, int, str)   # done, total, message
    ready    = pyqtSignal()
    error    = pyqtSignal(str)
    result   = pyqtSignal(str, object, object)  # matched_desc, betas, mesh_data


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class SemanticSearchGUI(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Populated by background loader
        self.tokenizer     = None
        self.embed_model   = None
        self.infer_device  = None
        self.descriptions  = None
        self.betas_list    = None
        self.db_embeddings = None   # (N, D) float32, L2-normalised
        self.smpl_model    = None

        self.signals = WorkerSignals()
        self.signals.progress.connect(self._on_progress)
        self.signals.ready.connect(self._on_ready)
        self.signals.error.connect(self._show_error)
        self.signals.result.connect(self._on_result)

        self._init_ui()
        threading.Thread(target=self._bg_load, daemon=True).start()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _init_ui(self):
        self.setWindowTitle("BodyShapeGPT — Semantic Search")
        self.resize(1120, 660)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # ---- Left panel ----
        left = QFrame()
        left.setFixedWidth(380)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(8)

        bold11  = QFont("Arial", 11, QFont.Bold)
        mono9   = QFont("Courier", 9)
        normal9 = QFont("Arial", 9)

        self.status_label = QLabel("Status: Initialising…")
        self.status_label.setFont(bold11)
        self.status_label.setWordWrap(True)
        ll.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        ll.addWidget(self.progress_bar)

        ll.addSpacing(6)

        ll.addWidget(QLabel("Body description:"))
        self.desc_input = QTextEdit()
        self.desc_input.setPlaceholderText(
            "e.g. 'tall person with broad shoulders and a slim waist'"
        )
        self.desc_input.setText("Average height person with broad shoulders")
        self.desc_input.setMaximumHeight(120)
        ll.addWidget(self.desc_input)

        self.search_btn = QPushButton("Search && Visualize")
        self.search_btn.setMinimumHeight(40)
        self.search_btn.setStyleSheet(
            "background-color: #2b78e4; color: white; font-weight: bold;"
        )
        self.search_btn.setEnabled(False)
        self.search_btn.clicked.connect(self._on_search_click)
        ll.addWidget(self.search_btn)

        ll.addSpacing(8)

        match_title = QLabel("Best match from dataset:")
        match_title.setFont(bold11)
        ll.addWidget(match_title)

        self.match_label = QLabel("—")
        self.match_label.setFont(normal9)
        self.match_label.setWordWrap(True)
        self.match_label.setStyleSheet(
            "background: #f0f4ff; border: 1px solid #c0c8e0;"
            "padding: 6px; border-radius: 4px;"
        )
        self.match_label.setMinimumHeight(70)
        ll.addWidget(self.match_label)

        ll.addSpacing(4)

        betas_title = QLabel("Shape params (betas):")
        betas_title.setFont(bold11)
        ll.addWidget(betas_title)

        self.params_label = QLabel("—")
        self.params_label.setFont(mono9)
        self.params_label.setWordWrap(True)
        ll.addWidget(self.params_label)

        ll.addStretch()

        # ---- Right panel (3-D viewer) ----
        right = QFrame()
        right.setStyleSheet("background-color: #1a1a1a;")
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)

        self.plotter = QtInteractor(right)
        self.plotter.set_background("#1a1a1a")
        rl.addWidget(self.plotter.interactor)

        main_layout.addWidget(left)
        main_layout.addWidget(right, stretch=1)

    # ------------------------------------------------------------------
    # Qt slots
    # ------------------------------------------------------------------

    def _update_status(self, text: str):
        self.status_label.setText(f"Status: {text}")

    def _show_error(self, msg: str):
        self.search_btn.setEnabled(True)
        self._update_status("Error.")
        QMessageBox.critical(self, "Error", msg)

    def _on_progress(self, done: int, total: int, msg: str):
        pct = int(done / total * 100) if total > 0 else 0
        self.progress_bar.setValue(pct)
        self._update_status(msg)

    def _on_ready(self):
        self.progress_bar.setValue(100)
        self._update_status("Ready — enter a description and click Search.")
        self.search_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Background: load model, dataset, embeddings
    # ------------------------------------------------------------------

    def _bg_load(self):
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.infer_device = device

            # 1. Tokenizer + model
            self.signals.progress.emit(0, 10, f"Loading {self.args.model}…")
            tok = AutoTokenizer.from_pretrained(self.args.model)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            mdl = AutoModel.from_pretrained(
                self.args.model,
                device_map=device,
                torch_dtype=torch.float16,
            )
            mdl.eval()
            self.tokenizer   = tok
            self.embed_model = mdl

            # 2. Dataset
            self.signals.progress.emit(1, 10, "Loading dataset…")
            descriptions, betas_list = load_dataset(self.args.dataset)
            self.descriptions = descriptions
            self.betas_list   = betas_list
            n = len(descriptions)

            # 3. Embeddings (cached)
            cache = self.args.embed_cache
            db_emb = None
            if os.path.exists(cache):
                self.signals.progress.emit(2, 10, "Loading cached embeddings…")
                loaded = np.load(cache)
                if loaded.shape[0] == n:
                    db_emb = loaded

            if db_emb is None:
                self.signals.progress.emit(
                    2, 10, f"Computing embeddings for {n} entries (one-time, ~1 min)…"
                )

                def _prog(done, total):
                    # Map [0..total] into progress range [2..9]
                    inner = int(done / total * 7)
                    self.signals.progress.emit(
                        2 + inner, 10,
                        f"Embedding dataset… {done}/{total}",
                    )

                db_emb = embed_texts(
                    descriptions, tok, mdl, device,
                    batch_size=64, progress_cb=_prog,
                )
                np.save(cache, db_emb)

            self.db_embeddings = db_emb  # (N, D), L2-normalised float32

            # 4. SMPL
            self.signals.progress.emit(9, 10, "Loading SMPL model…")
            self.smpl_model = smplx.create(
                model_path=_ROOT_DIR,
                model_type="smpl",
                gender=self.args.gender,
            )

            self.signals.ready.emit()

        except Exception as e:
            traceback.print_exc()
            self.signals.error.emit(str(e))

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _on_search_click(self):
        query = self.desc_input.toPlainText().strip()
        if not query:
            QMessageBox.warning(self, "Empty input", "Please enter a description.")
            return
        self.search_btn.setEnabled(False)
        self._update_status("Embedding query…")
        threading.Thread(
            target=self._bg_search, args=(query,), daemon=True
        ).start()

    def _bg_search(self, query: str):
        try:
            # Embed query (L2-normalised)
            q_emb = embed_texts(
                [query], self.tokenizer, self.embed_model,
                self.infer_device, batch_size=1,
            )  # (1, D)

            # Cosine similarity — both sides are already L2-normalised
            scores    = self.db_embeddings @ q_emb[0]   # (N,)
            best_idx  = int(np.argmax(scores))

            matched_desc = self.descriptions[best_idx]
            betas        = self.betas_list[best_idx]

            vertices, faces = self._run_smpl(betas)
            self.signals.result.emit(matched_desc, betas, (vertices, faces))

        except Exception as e:
            traceback.print_exc()
            self.signals.error.emit(str(e))

    def _run_smpl(self, betas: list):
        padded = (list(betas) + [0.0] * 10)[:10]
        beta_t = torch.tensor([padded], dtype=torch.float32)
        with torch.no_grad():
            out = self.smpl_model(betas=beta_t, return_verts=True)
        vertices = out.vertices.detach().cpu().numpy().squeeze()
        faces    = self.smpl_model.faces
        return vertices, faces

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _on_result(self, matched_desc: str, betas, mesh_data):
        self._update_status("Ready — enter a description and click Search.")
        self.search_btn.setEnabled(True)

        self.match_label.setText(matched_desc)
        beta_strs = ", ".join(f"{b:+.3f}" for b in betas)
        self.params_label.setText(f"[{beta_strs}]")

        vertices, faces = mesh_data
        self.plotter.clear_actors()

        faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).ravel()
        mesh = pv.PolyData(vertices, faces_pv)

        self.plotter.add_mesh(
            mesh,
            color="#ecdbcd",
            pbr=True,
            metallic=0.1,
            roughness=0.5,
            smooth_shading=True,
        )
        if hasattr(self.plotter, "enable_ssao"):
            self.plotter.enable_ssao(radius=0.5, bias=0.01)
        self.plotter.reset_camera()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="BodyShapeGPT — Semantic Database Search GUI"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Embedding model HF ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--dataset", default=DEFAULT_DATASET,
        help="Path to .jsonl dataset file",
    )
    parser.add_argument(
        "--embed-cache", default=DEFAULT_EMBED_CACHE, dest="embed_cache",
        help="Path to save/load pre-computed embeddings (.npy)",
    )
    parser.add_argument(
        "--gender", choices=["male", "female", "neutral"], default=DEFAULT_GENDER,
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    gui = SemanticSearchGUI(args)
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
