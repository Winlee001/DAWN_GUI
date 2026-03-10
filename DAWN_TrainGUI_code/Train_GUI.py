import os
import shutil
import sys
import shlex
import traceback
from datetime import datetime
from typing import Dict, Any, Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QComboBox,
    QCheckBox,
    QFileDialog,
    QPlainTextEdit,
    QFormLayout,
    QScrollArea,
    QMessageBox,
    QTabWidget,
)

from TrainRunner import TrainRunner
from TrainLogParser import TrainLogParser
from TrainMetrics import TrainMetrics
from GUI_Utils.SystemMonitor import MonitorThread

try:
    from omegaconf import OmegaConf
    OMEGA_AVAILABLE = True
except Exception:
    OMEGA_AVAILABLE = False

try:
    import yaml
    YAML_AVAILABLE = True
except Exception:
    YAML_AVAILABLE = False

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    MPL_AVAILABLE = True
except Exception:
    MPL_AVAILABLE = False


if MPL_AVAILABLE:
    class PlotCanvas(FigureCanvas):
        def __init__(self, title: str, parent=None):
            self.fig = Figure(figsize=(4, 3), dpi=100)
            self.ax = self.fig.add_subplot(111)
            super().__init__(self.fig)
            self.setParent(parent)
            self.ax.set_title(title)
            self.ax.grid(True, linestyle="--", alpha=0.3)

        def update_plot(self, x, y, xlabel: str, ylabel: str):
            self.ax.clear()
            self.ax.plot(x, y, marker="o", linewidth=1.5)
            self.ax.set_xlabel(xlabel)
            self.ax.set_ylabel(ylabel)
            self.ax.grid(True, linestyle="--", alpha=0.3)
            self.draw()


class TrainGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DAWN Train GUI")
        self.resize(1400, 800)

        if getattr(sys, "frozen", False):
            self.base_dir = os.path.dirname(sys.executable)
        else:
            self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.resource_dir = getattr(sys, "_MEIPASS", self.base_dir)
        self.workspace_dir = os.path.dirname(self.base_dir)
        self.DAWN_gray_dir = self.resolve_dawn_gray_dir()
        icon_path = os.path.join(self.resource_dir, "GUI_Utils", "icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.train_runner: Optional[TrainRunner] = None
        self.log_parser: Optional[TrainLogParser] = None
        self.log_timer = QTimer()
        self.log_timer.setInterval(1000)
        self.log_timer.timeout.connect(self.poll_logs)

        self.metrics = TrainMetrics()
        self.current_log_file = ""

        self.monitor_thread = None
        self.monitoring_active = False

        self.create_widgets()
        self.create_layout()
        self.start_monitoring()
        self.load_template_for_script()

    def resolve_dawn_gray_dir(self) -> str:
        # Prefer DAWN_gray located next to the packaged EXE.
        # This avoids accidentally binding to a developer workspace DAWN_gray.
        if getattr(sys, "frozen", False):
            candidates = [
                os.path.join(self.base_dir, "DAWN_gray"),
                os.path.join(self.workspace_dir, "DAWN_gray"),
                os.path.join(os.path.dirname(self.workspace_dir), "DAWN_gray"),
            ]
        else:
            candidates = [
                os.path.join(self.workspace_dir, "DAWN_gray"),
                os.path.join(self.base_dir, "DAWN_gray"),
                os.path.join(os.path.dirname(self.workspace_dir), "DAWN_gray"),
            ]
        for path in candidates:
            if os.path.isdir(path):
                return path
        return candidates[0]

    def resolve_python_exec_for_training(self) -> str:
        if not getattr(sys, "frozen", False):
            return sys.executable

        def _is_valid_python(path: str) -> bool:
            if not path:
                return False
            norm = os.path.normcase(os.path.normpath(path))
            # WindowsApps python is an app alias stub and often fails in subprocess (code 9009).
            if "windowsapps" in norm:
                return False
            return os.path.isfile(path)

        # In frozen mode, sys.executable points to DAWN_TrainGUI.exe.
        # Using it as the training interpreter would relaunch this GUI.
        env_python = os.environ.get("DAWN_TRAIN_PYTHON", "").strip().strip('"')
        if _is_valid_python(env_python):
            return env_python

        candidates = []
        conda_prefix = os.environ.get("CONDA_PREFIX", "").strip()
        if conda_prefix:
            candidates.append(os.path.join(conda_prefix, "python.exe"))

        user_home = os.path.expanduser("~")
        candidates.append(os.path.join(user_home, ".conda", "envs", "DAWN_code", "python.exe"))

        candidates.append(os.path.join(self.workspace_dir, ".venv", "Scripts", "python.exe"))
        candidates.append(os.path.join(self.workspace_dir, "venv", "Scripts", "python.exe"))

        for path in candidates:
            if _is_valid_python(path):
                return path

        which_python = shutil.which("python")
        if _is_valid_python(which_python):
            return which_python

        raise RuntimeError(
            "Cannot find a valid Python interpreter for training. "
            "Please set DAWN_TRAIN_PYTHON to a real python.exe path (not WindowsApps alias)."
        )

    def create_widgets(self):
        bfont = QFont()
        bfont.setBold(True)

        self.script_label = QLabel("Training Script")
        self.script_label.setFont(bfont)
        self.script_combo = QComboBox()
        self.script_combo.addItems([
            "gray_unpair_pretrain_mu.py",
            "gray_dualchan_pretrain_mu.py",
        ])
        self.script_combo.currentIndexChanged.connect(self.load_template_for_script)
        self.script_combo.currentIndexChanged.connect(self.update_dataset_input_mode)

        self.log_name_edit = QLineEdit("DAWN_gray")
        self.log_dir_edit = QLineEdit("ckpt")
        self.log_dir_btn = QPushButton("Browse")
        self.log_dir_btn.clicked.connect(self.browse_log_dir)

        self.trainset_edit = QLineEdit("imagenet_val")
        self.valset_edit = QLineEdit("bsd68")
        self.trainset_btn = QPushButton("Browse")
        self.valset_btn = QPushButton("Browse")
        self.trainset_btn.clicked.connect(lambda: self.browse_dataset_field(self.trainset_edit, "trainset"))
        self.valset_btn.clicked.connect(lambda: self.browse_dataset_field(self.valset_edit, "valset"))

        self.input_channel_edit = QLineEdit("3")
        self.output_channel_edit = QLineEdit("1")
        self.middle_channel_edit = QLineEdit("96")
        self.patch_size_edit = QLineEdit("96")
        self.batch_size_edit = QLineEdit("8")
        self.load_thread_edit = QLineEdit("4")
        self.epoch_edit = QLineEdit("90")
        self.steps_edit = QLineEdit("30,60,80")

        self.bsn_ver_combo = QComboBox()
        self.bsn_ver_combo.addItems([
            "dbsn",
            "dbsn_light",
            "dbsnl_fuse",
            "dbsn_fuse",
            "dbsn_fuseX",
            "dbsn_centrblind",
            "2stage_DBSN",
            "2stage_DBSNL",
            "DTB_DBSN",
            "DBSN_AttCB",
            "dbsnl_centrblind",
            "dbsn_DC",
        ])
        self.bsn_ver_combo.setCurrentText("dbsn_centrblind")

        self.loss_choice_combo = QComboBox()
        self.loss_choice_combo.addItems([
            "L2",
            "L2_video",
            "L2_video_AC",
            "L2_video_WeiImage",
            "DBSNL",
            "MAPL",
            "DBSN_video",
        ])
        self.loss_choice_combo.setCurrentText("L2_video")

        self.weight_combo = QComboBox()
        self.weight_combo.addItems(["One", "all"])

        self.lr_dbsn_edit = QLineEdit("0.0003")
        self.decay_rate_edit = QLineEdit("0.1")
        self.data_rate_edit = QLineEdit("1.0")
        self.repeat_edit = QLineEdit("1")

        self.blindspot_type_combo = QComboBox()
        self.blindspot_type_combo.addItems(["Trimmed", "Mask", "MulMask", "Mask3D"])
        self.blindspot_type_combo.setCurrentText("Mask3D")
        self.blindspot_bias_combo = QComboBox()
        self.blindspot_bias_combo.addItems(["True", "False"])
        self.blindspot_bias_combo.setCurrentText("True")

        self.mask_shape_combo = QComboBox()
        self.mask_shape_combo.addItems(["o", "x", "+"])

        self.resume_combo = QComboBox()
        self.resume_combo.addItems(["new", "continue"])
        self.last_ckpt_edit = QLineEdit("")
        self.last_ckpt_btn = QPushButton("Browse")
        self.last_ckpt_btn.clicked.connect(self.browse_last_ckpt)

        self.ct_desc_edit = QLineEdit("")
        self.device_ids_edit = QLineEdit("all")

        self.dynamic_load_check = QCheckBox("dynamic_load")
        self.no_flip_check = QCheckBox("no_flip")
        self.frame_shuffle_check = QCheckBox("Frame_shuffle")
        self.dynamic_load_check.setChecked(True)

        self.load_yaml_btn = QPushButton("Load YAML")
        self.save_yaml_btn = QPushButton("Save YAML")
        self.load_yaml_btn.clicked.connect(self.load_yaml)
        self.save_yaml_btn.clicked.connect(self.save_yaml)

        self.import_sh_btn = QPushButton("Import .sh")
        self.import_sh_btn.clicked.connect(self.import_sh)

        self.start_btn = QPushButton("Start Training")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_training)
        self.stop_btn.clicked.connect(self.stop_training)

        self.output_log = QPlainTextEdit()
        self.output_log.setReadOnly(True)

        self.epoch_label = QLabel("Epoch: --")
        self.step_label = QLabel("Step: --")
        self.loss_label = QLabel("Loss: --")
        self.lr_label = QLabel("LR: --")

        self.monitor_label = QLabel("System Monitor")
        self.monitor_label.setFont(bfont)
        self.cpu_monitor_label = QLabel("CPU: -- % | Memory: -- %")
        self.gpu_monitor_label = QLabel("GPU: Not monitoring")
        self.gpu_monitor_label.setWordWrap(True)

        if MPL_AVAILABLE:
            self.loss_plot = PlotCanvas("Avg Loss")
            self.lr_plot = PlotCanvas("Learning Rate")
        else:
            self.loss_plot = QLabel("Matplotlib is not available")
            self.lr_plot = QLabel("Matplotlib is not available")

        self.field_map = {
            "script": self.script_combo,
            "log_name": self.log_name_edit,
            "log_dir": self.log_dir_edit,
            "trainset": self.trainset_edit,
            "valset": self.valset_edit,
            "input_channel": self.input_channel_edit,
            "output_channel": self.output_channel_edit,
            "middle_channel": self.middle_channel_edit,
            "patch_size": self.patch_size_edit,
            "batch_size": self.batch_size_edit,
            "load_thread": self.load_thread_edit,
            "epoch": self.epoch_edit,
            "steps": self.steps_edit,
            "bsn_ver": self.bsn_ver_combo,
            "loss_choice": self.loss_choice_combo,
            "weight": self.weight_combo,
            "lr_dbsn": self.lr_dbsn_edit,
            "decay_rate": self.decay_rate_edit,
            "data_rate": self.data_rate_edit,
            "repeat": self.repeat_edit,
            "blindspot_conv_type": self.blindspot_type_combo,
            "blindspot_conv_bias": self.blindspot_bias_combo,
            "mask_shape": self.mask_shape_combo,
            "resume": self.resume_combo,
            "last_ckpt": self.last_ckpt_edit,
            "Run_description": self.ct_desc_edit,
            "device_ids": self.device_ids_edit,
            "dynamic_load": self.dynamic_load_check,
            "no_flip": self.no_flip_check,
            "frame_shuffle": self.frame_shuffle_check,
        }

    def create_layout(self):
        root = QWidget()
        main_layout = QHBoxLayout()

        form_widget = QWidget()
        form_layout = QFormLayout()

        trainset_layout = QHBoxLayout()
        trainset_layout.addWidget(self.trainset_edit)
        trainset_layout.addWidget(self.trainset_btn)

        valset_layout = QHBoxLayout()
        valset_layout.addWidget(self.valset_edit)
        valset_layout.addWidget(self.valset_btn)

        log_dir_layout = QHBoxLayout()
        log_dir_layout.addWidget(self.log_dir_edit)
        log_dir_layout.addWidget(self.log_dir_btn)

        last_ckpt_layout = QHBoxLayout()
        last_ckpt_layout.addWidget(self.last_ckpt_edit)
        last_ckpt_layout.addWidget(self.last_ckpt_btn)

        form_layout.addRow(self.script_label, self.script_combo)
        form_layout.addRow(QLabel("log_name"), self.log_name_edit)
        form_layout.addRow(QLabel("log_dir"), log_dir_layout)
        form_layout.addRow(QLabel("trainset"), trainset_layout)
        form_layout.addRow(QLabel("valset"), valset_layout)
        form_layout.addRow(QLabel("input_channel"), self.input_channel_edit)
        form_layout.addRow(QLabel("output_channel"), self.output_channel_edit)
        form_layout.addRow(QLabel("middle_channel"), self.middle_channel_edit)
        form_layout.addRow(QLabel("patch_size"), self.patch_size_edit)
        form_layout.addRow(QLabel("batch_size"), self.batch_size_edit)
        form_layout.addRow(QLabel("load_thread"), self.load_thread_edit)
        form_layout.addRow(QLabel("epoch"), self.epoch_edit)
        form_layout.addRow(QLabel("steps"), self.steps_edit)
        form_layout.addRow(QLabel("bsn_ver"), self.bsn_ver_combo)
        form_layout.addRow(QLabel("loss_choice"), self.loss_choice_combo)
        form_layout.addRow(QLabel("weight"), self.weight_combo)
        form_layout.addRow(QLabel("lr_dbsn"), self.lr_dbsn_edit)
        form_layout.addRow(QLabel("decay_rate"), self.decay_rate_edit)
        form_layout.addRow(QLabel("data_rate"), self.data_rate_edit)
        form_layout.addRow(QLabel("repeat"), self.repeat_edit)
        form_layout.addRow(QLabel("blindspot_conv_type"), self.blindspot_type_combo)
        form_layout.addRow(QLabel("blindspot_conv_bias"), self.blindspot_bias_combo)
        form_layout.addRow(QLabel("mask_shape"), self.mask_shape_combo)
        form_layout.addRow(QLabel("resume"), self.resume_combo)
        form_layout.addRow(QLabel("last_ckpt"), last_ckpt_layout)
        form_layout.addRow(QLabel("Run_description"), self.ct_desc_edit)
        form_layout.addRow(QLabel("device_ids"), self.device_ids_edit)

        checks_layout = QVBoxLayout()
        checks_layout.addWidget(self.dynamic_load_check)
        checks_layout.addWidget(self.no_flip_check)
        checks_layout.addWidget(self.frame_shuffle_check)
        form_layout.addRow(QLabel("flags"), checks_layout)

        button_row = QHBoxLayout()
        button_row.addWidget(self.load_yaml_btn)
        button_row.addWidget(self.save_yaml_btn)
        button_row.addWidget(self.import_sh_btn)
        form_layout.addRow(QLabel("config"), button_row)

        form_widget.setLayout(form_layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(form_widget)

        left_layout = QVBoxLayout()
        left_layout.addWidget(scroll)
        left_container = QWidget()
        left_container.setLayout(left_layout)

        right_layout = QVBoxLayout()

        run_layout = QHBoxLayout()
        run_layout.addWidget(self.start_btn)
        run_layout.addWidget(self.stop_btn)
        right_layout.addLayout(run_layout)

        status_layout = QHBoxLayout()
        status_layout.addWidget(self.epoch_label)
        status_layout.addWidget(self.step_label)
        status_layout.addWidget(self.loss_label)
        status_layout.addWidget(self.lr_label)
        right_layout.addLayout(status_layout)

        monitor_container = QWidget()
        monitor_layout = QVBoxLayout()
        monitor_layout.addWidget(self.monitor_label)
        monitor_layout.addWidget(self.cpu_monitor_label)
        monitor_layout.addWidget(self.gpu_monitor_label)
        monitor_container.setLayout(monitor_layout)
        monitor_container.setStyleSheet("QWidget { background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 5px; }")
        right_layout.addWidget(monitor_container)

        tab_widget = QTabWidget()
        tab_widget.addTab(self.output_log, "Logs")
        tab_widget.addTab(self.loss_plot, "Loss")
        tab_widget.addTab(self.lr_plot, "LR")
        right_layout.addWidget(tab_widget)

        right_container = QWidget()
        right_container.setLayout(right_layout)

        main_layout.addWidget(left_container, 2)
        main_layout.addWidget(right_container, 3)

        root.setLayout(main_layout)
        self.setCentralWidget(root)

    def browse_log_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Folder", self.workspace_dir)
        if directory:
            self.log_dir_edit.setText(directory)

    def browse_last_ckpt(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Checkpoint", self.workspace_dir, "Checkpoint (*.pth)")
        if file_path:
            self.last_ckpt_edit.setText(file_path)

    def load_yaml(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load YAML", self.workspace_dir, "YAML (*.yaml *.yml)")
        if not file_path:
            return
        data = self.read_yaml(file_path)
        if data is None:
            self.show_error("Failed to read YAML file.")
            return
        self.apply_config(data)

    def load_template_for_script(self):
        script_name = self.script_combo.currentText().strip()
        template_name = None
        if script_name == "gray_unpair_pretrain_mu.py":
            template_name = "DAWN_SingleChannel_train_template.yaml"
        elif script_name == "gray_dualchan_pretrain_mu.py":
            template_name = "DAWN_DualChannel_train_template.yaml"

        if not template_name:
            return

        template_path = os.path.join(self.resource_dir, "config_templates", template_name)
        if not os.path.exists(template_path):
            # Fallback for script mode or custom launch layouts
            fallback = os.path.join(self.base_dir, "config_templates", template_name)
            if os.path.exists(fallback):
                template_path = fallback
            else:
                self.append_log(f"Template not found: {template_name}")
                return
        data = self.read_yaml(template_path)
        if data is None:
            self.append_log(f"Failed to read template: {template_path}")
            return
        self.apply_config(data)
        self.update_dataset_input_mode()

    def is_dual_script(self):
        return self.script_combo.currentText().strip() == "gray_dualchan_pretrain_mu.py"

    def update_dataset_input_mode(self):
        if self.is_dual_script():
            self.trainset_edit.setPlaceholderText("Dual mode: set two values separated by comma")
            self.valset_edit.setPlaceholderText("Dual mode: set two values separated by comma")
            self.trainset_btn.setToolTip("Select folders in two clicks: first path1, then path2")
            self.valset_btn.setToolTip("Select folders in two clicks: first path1, then path2")
        else:
            train_tokens = [x.strip() for x in self.trainset_edit.text().split(",") if x.strip()]
            val_tokens = [x.strip() for x in self.valset_edit.text().split(",") if x.strip()]
            if len(train_tokens) >= 1:
                self.trainset_edit.setText(train_tokens[0])
            if len(val_tokens) >= 1:
                self.valset_edit.setText(val_tokens[0])
            self.trainset_edit.setPlaceholderText("Single mode: set one value")
            self.valset_edit.setPlaceholderText("Single mode: set one value")
            self.trainset_btn.setToolTip("Select one folder for trainset")
            self.valset_btn.setToolTip("Select one folder for valset")

    def browse_dataset_field(self, target_edit: QLineEdit, field_name: str):
        selected = QFileDialog.getExistingDirectory(self, f"Select {field_name} folder", self.workspace_dir)
        if not selected:
            return
        if not self.is_dual_script():
            target_edit.setText(selected)
            return

        # Dual mode: support incremental two-click selection.
        # 1st click -> path1, 2nd click -> path1,path2
        existing = [x.strip() for x in target_edit.text().split(",") if x.strip()]
        if len(existing) == 0:
            target_edit.setText(selected)
        elif len(existing) == 1:
            target_edit.setText(f"{existing[0]},{selected}")
        else:
            # If already has two values, start a new pair from current selection.
            target_edit.setText(selected)

    def save_yaml(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save YAML", self.workspace_dir, "YAML (*.yaml *.yml)")
        if not file_path:
            return
        data = self.collect_config()
        self.write_yaml(file_path, data)

    def import_sh(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Import .sh", self.workspace_dir, "Shell Script (*.sh)")
        if not file_path:
            return
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        tokens = shlex.split(content.replace("\\\n", " "))
        args = {}
        key = None
        for token in tokens:
            if token.startswith("--"):
                key = token[2:]
                args[key] = True
            elif key:
                args[key] = token
                key = None
        self.apply_config(args)

    def collect_config(self) -> Dict[str, Any]:
        config = {}
        for key, widget in self.field_map.items():
            if isinstance(widget, QLineEdit):
                config[key] = widget.text().strip()
            elif isinstance(widget, QComboBox):
                config[key] = widget.currentText().strip()
            elif isinstance(widget, QCheckBox):
                config[key] = widget.isChecked()
        return config

    def apply_config(self, data: Dict[str, Any]):
        alias_map = {
            "Loss_choice": "loss_choice",
            "Frame_shuffle": "frame_shuffle",
            "Run_description": "Run_description",
        }
        for key, value in data.items():
            mapped_key = alias_map.get(key, key)
            if mapped_key not in self.field_map:
                continue
            widget = self.field_map[mapped_key]
            if isinstance(widget, QLineEdit):
                widget.setText(str(value))
            elif isinstance(widget, QComboBox):
                widget.setCurrentText(str(value))
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))

    def build_command(self):
        config = self.collect_config()
        script_name = config.get("script") or self.script_combo.currentText()
        script_path = os.path.join(self.DAWN_gray_dir, script_name)
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Script not found: {script_path}")

        python_exec = self.resolve_python_exec_for_training()
        cmd = [python_exec, "-u", script_path]

        def add_arg(name: str, value: Any):
            if value is None or value == "":
                return
            cmd.append(f"--{name}")
            cmd.append(str(value))

        add_arg("log_name", config.get("log_name"))
        log_dir_raw = config.get("log_dir")
        if log_dir_raw and not os.path.isabs(log_dir_raw):
            log_dir_for_train = os.path.normpath(os.path.join(self.DAWN_gray_dir, log_dir_raw))
        else:
            log_dir_for_train = log_dir_raw
        add_arg("log_dir", log_dir_for_train)
        add_arg("trainset", config.get("trainset"))
        add_arg("valset", config.get("valset"))
        add_arg("input_channel", config.get("input_channel"))
        add_arg("output_channel", config.get("output_channel"))
        add_arg("middle_channel", config.get("middle_channel"))
        add_arg("patch_size", config.get("patch_size"))
        add_arg("batch_size", config.get("batch_size"))
        add_arg("load_thread", config.get("load_thread"))
        add_arg("epoch", config.get("epoch"))
        add_arg("steps", config.get("steps"))
        add_arg("bsn_ver", config.get("bsn_ver"))
        add_arg("Loss_choice", config.get("loss_choice"))
        add_arg("weight", config.get("weight"))
        add_arg("lr_dbsn", config.get("lr_dbsn"))
        add_arg("decay_rate", config.get("decay_rate"))
        add_arg("data_rate", config.get("data_rate"))
        add_arg("repeat", config.get("repeat"))
        add_arg("blindspot_conv_type", config.get("blindspot_conv_type"))
        add_arg("blindspot_conv_bias", config.get("blindspot_conv_bias"))
        add_arg("mask_shape", config.get("mask_shape"))
        add_arg("resume", config.get("resume"))
        if (config.get("resume") or "").lower() == "continue":
            add_arg("last_ckpt", config.get("last_ckpt"))
        add_arg("Run_description", config.get("Run_description"))
        add_arg("device_ids", config.get("device_ids"))

        if config.get("dynamic_load"):
            cmd.append("--dynamic_load")
            cmd.append("True")
        if config.get("no_flip"):
            cmd.append("--no_flip")
        if config.get("frame_shuffle"):
            cmd.append("--Frame_shuffle")
            cmd.append("True")

        return cmd

    def start_training(self):
        if self.train_runner and self.train_runner.isRunning():
            self.show_error("Training is already running.")
            return
        train_tokens = [x.strip() for x in self.trainset_edit.text().split(",") if x.strip()]
        val_tokens = [x.strip() for x in self.valset_edit.text().split(",") if x.strip()]
        if self.is_dual_script():
            if len(train_tokens) != 2 or len(val_tokens) != 2:
                self.show_error("Dual-channel mode requires exactly two values for trainset and valset.")
                return
        else:
            if len(train_tokens) != 1 or len(val_tokens) != 1:
                self.show_error("Single-channel mode requires exactly one value for trainset and valset.")
                return
        if not self.ct_desc_edit.text().strip():
            self.show_error("Run_description is required.")
            return
        if self.resume_combo.currentText() == "continue" and not self.last_ckpt_edit.text().strip():
            self.show_error("resume=continue requires last_ckpt.")
            return
        try:
            cmd = self.build_command()
        except Exception as e:
            self.show_error(str(e))
            return

        self.metrics = TrainMetrics()
        self.output_log.clear()

        self.current_log_file = self.get_log_file_path()
        self.log_parser = TrainLogParser(self.current_log_file)

        self.train_runner = TrainRunner(cmd, cwd=self.DAWN_gray_dir)
        self.train_runner.outputLine.connect(self.append_log)
        self.train_runner.errorLine.connect(self.append_log)
        self.train_runner.lrParsed.connect(self.on_lr_parsed)
        self.train_runner.finishedProcess.connect(self.on_process_finished)
        self.train_runner.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.log_timer.start()

        self.append_log("Starting training...")
        self.append_log("Command: " + " ".join(cmd))
        self.append_log(f"Log file: {self.current_log_file}")

    def stop_training(self):
        if self.train_runner:
            self.train_runner.request_stop()
        self.log_timer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.append_log("Training stop requested.")

    def on_process_finished(self, code: int):
        self.log_timer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.append_log(f"Training finished with code: {code}")

    def append_log(self, text: str):
        self.output_log.appendPlainText(text)

    def on_lr_parsed(self, lr: float):
        self.metrics.append_lr(lr)
        self.lr_label.setText(f"LR: {lr:.6f}")
        self.update_lr_plot()

    def poll_logs(self):
        if not self.log_parser:
            return
        for line in self.log_parser.read_new_lines():
            dataset_size = self.log_parser.parse_dataset_size(line)
            if dataset_size is not None:
                self.metrics.set_dataset_size(dataset_size)

            epoch_info = self.log_parser.parse_epoch_loss(line)
            if epoch_info:
                epoch, total, best, loss = epoch_info
                step = None
                if self.metrics.dataset_size:
                    step = (epoch + 1) * int(self.metrics.dataset_size)
                self.metrics.append_epoch(epoch, loss, step=step)
                self.epoch_label.setText(f"Epoch: {epoch}/{total} (best: {best})")
                if step is not None:
                    self.step_label.setText(f"Step: {step}")
                self.loss_label.setText(f"Loss: {loss:.6f}")
                self.update_loss_plot()

    def update_loss_plot(self):
        if not MPL_AVAILABLE:
            return
        if not self.metrics.epochs:
            return
        self.loss_plot.update_plot(self.metrics.epochs, self.metrics.avg_loss, "Epoch", "Avg Loss")

    def update_lr_plot(self):
        if not MPL_AVAILABLE:
            return
        if not self.metrics.lr:
            return
        x = list(range(len(self.metrics.lr)))
        self.lr_plot.update_plot(x, self.metrics.lr, "Step (lr logs)", "LR")

    def get_log_file_path(self) -> str:
        config = self.collect_config()
        log_dir = config.get("log_dir") or self.log_dir_edit.text().strip()
        if log_dir and not os.path.isabs(log_dir):
            log_dir = os.path.normpath(os.path.join(self.DAWN_gray_dir, log_dir))
        log_name = config.get("log_name") or "DAWN_gray"
        noise_type = "Noisy_data"
        run_desc = config.get("Run_description") or "None"
        # Must match DAWN_gray training scripts:
        # checkpoint_dir = args.save_prefix + '_' + noise_level
        # save_prefix = log_name + '_' + noise_type
        # noise_level = Run_description (when noise_type == Noisy_data)
        checkpoint_dir = f"{log_name}_{noise_type}_{run_desc}"
        return os.path.join(log_dir, checkpoint_dir + "_log.txt")

    def read_yaml(self, file_path: str) -> Optional[Dict[str, Any]]:
        if OMEGA_AVAILABLE:
            try:
                conf = OmegaConf.load(file_path)
                return dict(conf)
            except Exception:
                pass
        if YAML_AVAILABLE:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f)
            except Exception:
                return None
        return None

    def write_yaml(self, file_path: str, data: Dict[str, Any]):
        if OMEGA_AVAILABLE:
            try:
                OmegaConf.save(config=OmegaConf.create(data), f=file_path)
                return
            except Exception:
                pass
        if YAML_AVAILABLE:
            with open(file_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
            return
        self.show_error("No YAML backend available.")

    def show_error(self, message: str):
        msg = QMessageBox(self)
        msg.setWindowTitle("Error")
        msg.setText(message)
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.exec()

    def start_monitoring(self):
        if self.monitoring_active:
            return
        try:
            self.monitor_thread = MonitorThread(self, interval=1.0)
            self.monitor_thread.metricsUpdated.connect(self.update_monitor_ui)
            self.monitor_thread.start()
            self.monitoring_active = True
        except Exception as e:
            self.append_log(f"Monitor disabled: {e}")

    def update_monitor_ui(self, metrics):
        self.cpu_monitor_label.setText(
            f"CPU: {metrics.cpu_percent:.1f} % | Memory: {metrics.cpu_memory_percent:.1f} %"
        )
        if metrics.gpu_count == 0:
            self.gpu_monitor_label.setText("GPU: Not available")
        else:
            parts = []
            for g in metrics.gpu_metrics:
                parts.append(
                    f"{g['device_name']} | GPU: {g['gpu_util']}% | Mem: {g['memory_percent']:.1f}%"
                )
            self.gpu_monitor_label.setText("\n".join(parts))


if __name__ == "__main__":
    def _runtime_log_path():
        if getattr(sys, "frozen", False):
            base = os.path.dirname(sys.executable)
        else:
            base = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base, "DAWN_TrainGUI_crash.log")

    def _excepthook(exc_type, exc_value, exc_tb):
        log_path = _runtime_log_path()
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Unhandled Exception\n")
            f.write("".join(traceback.format_exception(exc_type, exc_value, exc_tb)))
        try:
            QMessageBox.critical(
                None,
                "DAWN Train GUI Error",
                f"程序发生异常并已退出。\n错误日志已保存到:\n{log_path}",
            )
        except Exception:
            pass

    sys.excepthook = _excepthook

    app = QApplication(sys.argv)
    window = TrainGUI()
    window.show()
    sys.exit(app.exec())
