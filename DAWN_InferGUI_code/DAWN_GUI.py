import sys
import os
from os.path import join as opj
from os.path import dirname as opd

import tifffile
import numpy as np
import time
import traceback

from omegaconf import OmegaConf
from PyQt6.QtCore import QUrl
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent, QStandardItemModel, QStandardItem, QPixmap, QImage, QDoubleValidator, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QLabel,
    QPushButton,
    QMenu,
    QTreeView,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
    QSlider,
    QLineEdit,
    QProgressBar,
    QMessageBox,
    QCheckBox,
    QRadioButton,
    QButtonGroup,
    QPlainTextEdit,
)
from PyQt6.QtCore import QEvent
from PyQt6.QtGui import QIcon
import torch
import subprocess

from GUI_Utils.Utils import load_tiff_seq, ReconstructThread
from GUI_Utils.SystemMonitor import MonitorThread


class VideoReconstructionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("DAWN GUI")
        # 设置图标
        self.setWindowIcon(QIcon("./GUI_Utils/icon.png"))

        self.output_folder = "./outputs"
        self.ch_a_numpy = None
        self.ch_b_numpy = None
        self.single_ch_numpy = None
        self.linear_gain = 1
        self.enhance_a = True
        self.enhance_b = True
        self.mode = "dual"  # "single" or "dual"
        self.ch_a_paths = []
        self.ch_b_paths = []
        self.single_paths = []
        self.ch_a_rel_paths = []
        self.ch_b_rel_paths = []
        self.single_rel_paths = []
        self.ch_a_is_dir = False
        self.ch_b_is_dir = False
        self.single_is_dir = False
        self.reconstruct_thread = None

        self.opt_path = "./GUI_Utils/Model/DAWN_CKPT/dbsn_gray_CT_data_CT_datadbsn3DCBF_unetskip_Dv7SWNA_31_0500_AP2/DBSN_fusion_Model.yaml"
        self.model_path = "./GUI_Utils/Model/DAWN_CKPT/dbsn_gray_CT_data_CT_datadbsn3DCBF_unetskip_Dv7SWNA_31_0500_AP2/dbsn_gray_CT_data_ckpt_e39.pth"  # Can be .pth or .pkl

        print("CUDA Device Count: " + str(torch.cuda.device_count()))
        self.device_list = [f"cuda:{i}" for i in range(torch.cuda.device_count())] + ["cpu"]
        self.selected_device = self.device_list[0]

        # Initialize monitoring thread
        self.monitor_thread = None
        self.monitoring_active = False

        self.create_widgets()
        self.create_layout()
        
        # 启动系统监控，让它一直运行
        self.start_monitoring()

        try:
            self.ch_a_numpy = self.loadAndSetInputPath("./GUI_Utils/Test_Img/EMCCD/EMCCD_0723/acc/500mW/s1502_0004/s1502_0004_1.tif", "a")
            self.ch_b_numpy = self.loadAndSetInputPath("./GUI_Utils/Test_Img/EMCCD/EMCCD_0723/don/500mW/s1502_0004/s1502_0004_1.tif", "b")
        except:
            self.ch_a_numpy = np.ones([50, 200, 200]) * 0
            self.ch_b_numpy = np.ones([50, 200, 200]) * 0
            print(f"Preload test images failed! Please manually load the input images.")
        self.ch_a_out_numpy = np.ones_like(self.ch_a_numpy) * 0 if self.ch_a_numpy is not None else None
        self.ch_b_out_numpy = np.ones_like(self.ch_b_numpy) * 0 if self.ch_b_numpy is not None else None
        self.single_ch_out_numpy = None

        if len(self.device_list) == 1 and self.device_list[0] == "cpu":
            # 弹出提示框
            msg_box = QMessageBox()
            msg_box.setWindowTitle("Warning!")
            msg_box.setText(
                f"GPU is not available! Please check your CUDA installation if you have a GPU. \nDAWN will run on CPU. The reconstruction will be extremely slow. \nWe recommend using GPU."
            )
            msg_box.setIcon(QMessageBox.Icon.Information)
            msg_box.addButton(QMessageBox.StandardButton.Ok)
            msg_box.exec()

    def create_widgets(self):
        def create_video_viewer(name, size):
            video_viewer = QLabel(name)
            # video_viewer.setPixmap(QPixmap("GUI_Utils/test.png"))
            video_viewer.setFixedSize(*size)
            video_viewer.setAcceptDrops(True)
            video_viewer.installEventFilter(self)
            # 设置图像居中
            video_viewer.setAlignment(Qt.AlignmentFlag.AlignCenter)
            video_viewer.setScaledContents(True)
            return video_viewer

        def create_view_slider(name, size, value_changed_func):
            view_slider = QSlider(Qt.Orientation.Horizontal, self)
            view_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            view_slider.setFixedSize(*size)
            view_slider.valueChanged[int].connect(value_changed_func)
            return view_slider

        bfont = QFont()
        bfont.setBold(True)

        self.input_label1 = QLabel("Acceptor dark")
        self.input_label1.setFont(bfont)
        self.input_path1 = QLineEdit()
        self.input_path1.setPlaceholderText("Acceptor path (file or folder)")
        self.input_path1.returnPressed.connect(lambda: self.set_input_path_from_edit("a"))
        self.input_browse1 = QPushButton("Browse")
        self.input_browse1.setMenu(self.build_input_browse_menu("a"))
        self.input_label2 = QLabel("Donor dark")
        self.input_label2.setFont(bfont)
        self.input_path2 = QLineEdit()
        self.input_path2.setPlaceholderText("Donor path (file or folder)")
        self.input_path2.returnPressed.connect(lambda: self.set_input_path_from_edit("b"))
        self.input_browse2 = QPushButton("Browse")
        self.input_browse2.setMenu(self.build_input_browse_menu("b"))
        self.single_input_label = QLabel("Single channel input")
        self.single_input_label.setFont(bfont)
        self.single_input_path = QLineEdit()
        self.single_input_path.setPlaceholderText("Single channel path (file or folder)")
        self.single_input_path.returnPressed.connect(lambda: self.set_input_path_from_edit("single"))
        self.single_input_browse = QPushButton("Browse")
        self.single_input_browse.setMenu(self.build_input_browse_menu("single"))
        self.output_label1 = QLabel("Acceptor recon.")
        self.output_label1.setFont(bfont)
        self.output_label2 = QLabel("Donor recon.")
        self.output_label2.setFont(bfont)
        self.single_output_label = QLabel("Single channel recon.")
        self.single_output_label.setFont(bfont)

        self.input_video1 = create_video_viewer("Acceptor dark", (300, 300))
        self.in1_sld = create_view_slider("Acceptor dark", (300, 20), self.changeValueIn1)

        self.input_video2 = create_video_viewer("Donor dark", (300, 300))
        self.in2_sld = create_view_slider("Donor dark", (300, 20), self.changeValueIn2)

        self.single_input_video = create_video_viewer("Single channel input", (300, 300))
        self.single_in_sld = create_view_slider("Single channel input", (300, 20), self.changeValueSingleIn)

        self.output_video1 = create_video_viewer("Acceptor recon.", (300, 300))
        self.out1_sld = create_view_slider("Acceptor recon.", (300, 20), self.changeValueOut1)

        self.output_video2 = create_video_viewer("Donor recon.", (300, 300))
        self.out2_sld = create_view_slider("Donor recon.", (300, 20), self.changeValueOut2)

        self.single_output_video = create_video_viewer("Single channel recon.", (300, 300))
        self.single_out_sld = create_view_slider("Single channel recon.", (300, 20), self.changeValueSingleOut)

        self.instructions_label = QLabel(
            'Instructions: \n1. Select single or dual channel mode. \n2. Drag and drop input videos, or type a file/folder path and press Enter. \n3. Choose the model file, config file and output folder. \n4. Set the linear gain if needed. \n5. Select the device to use (GPU is recommended). \n6. Click the "Reconstruct" button and wait for the completion. The enhanced videos will be saved to output folder.'
        )
        self.instructions_label.setWordWrap(True)

        self.output_label = QLabel("Output Video")
        self.output_label.setFont(bfont)
        # 使用 QPlainTextEdit 替代 QLabel，支持滚动查看完整路径
        self.output_path_show = QPlainTextEdit("Path: ./outputs")
        self.output_path_show.setFixedHeight(75)
        self.output_path_show.setReadOnly(True)  # 设置为只读
        self.output_path_show.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)  # 不自动换行，使用水平滚动
        self.output_path_show.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)  # 需要时显示垂直滚动条
        self.output_path_show.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)  # 需要时显示水平滚动条
        self.browse_output_button = QPushButton("Browse Output Folder")
        self.browse_output_button.clicked.connect(self.browse_output_folder)

        self.device_label = QLabel("Select Device")
        self.device_label.setFont(bfont)
        self.device_tree = QTreeView()
        self.device_model = QStandardItemModel()
        for device in self.device_list:
            device_name = torch.cuda.get_device_name(device) if device != "cpu" else "CPU"
            item = QStandardItem(f"{device_name} ({device})")
            self.device_model.appendRow(item)
        self.device_tree.setModel(self.device_model)
        self.device_tree.clicked.connect(self.device_selected)
        # 预先选中第一个
        self.device_tree.setCurrentIndex(self.device_model.index(0, 0))
        # 设置一个输入框用来设置linear gain

        self.linear_gain_label = QLabel("Linear Gain")
        self.linear_gain_label.setFont(bfont)
        self.linear_gain_input = QLineEdit("1")
        self.linear_gain_input.setValidator(QDoubleValidator())
        self.linear_gain_input.textEdited.connect(lambda x: setattr(self, "linear_gain", float(x) if x != "" else 0))
        self.linear_gain_set_button = QPushButton("Set Linear Gain")
        self.linear_gain_set_button.clicked.connect(self.set_linear_gain)

        self.model_path_label = QLabel("Model File")
        self.model_path_label.setFont(bfont)
        # 使用 QPlainTextEdit 替代 QLabel，支持滚动查看完整路径
        self.model_path_show = QPlainTextEdit("Path: ./GUI_Utils/Model/DAWN_CKPT/dbsn_gray_CT_data_CT_datadbsn3DCBF_unetskip_Dv7SWNA_31_0500_AP2/dbsn_gray_CT_data_ckpt_e39.pth (supports .pth and .pkl)")
        self.model_path_show.setFixedHeight(75)
        self.model_path_show.setReadOnly(True)  # 设置为只读
        self.model_path_show.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)  # 不自动换行，使用水平滚动
        self.model_path_show.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)  # 需要时显示垂直滚动条
        self.model_path_show.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)  # 需要时显示水平滚动条
        self.model_path_show.setAcceptDrops(True)
        self.model_path_show.installEventFilter(self)
        self.browse_model_button = QPushButton("Browse model file")
        self.browse_model_button.clicked.connect(self.browse_model)
        
        self.opt_path_label = QLabel("Config File (YAML)")
        self.opt_path_label.setFont(bfont)
        # 使用 QPlainTextEdit 替代 QLabel，支持滚动查看完整路径 "C:\Users\Admin\Desktop\MUFFLE\Cursor_implement\DAWN_Code\GUI_Utils\Model\CKPT\dbsn_gray_CT_data_CT_datadbsn3DCBF_unetskip_Dv7SWNA_31_0500_AP2\dbsn_gray_CT_data_ckpt_e39.pth"
        self.opt_path_show = QPlainTextEdit("Path: ./GUI_Utils/Model/DAWN_CKPT/dbsn_gray_CT_data_CT_datadbsn3DCBF_unetskip_Dv7SWNA_31_0500_AP2/DBSN_fusion_Model.yaml")
        self.opt_path_show.setFixedHeight(75)
        self.opt_path_show.setReadOnly(True)  # 设置为只读
        self.opt_path_show.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)  # 不自动换行，使用水平滚动
        self.opt_path_show.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)  # 需要时显示垂直滚动条
        self.opt_path_show.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)  # 需要时显示水平滚动条
        self.browse_opt_button = QPushButton("Browse Config file (.yaml)")
        self.browse_opt_button.clicked.connect(self.browse_config)

        # Mode selection (single/dual channel)
        self.mode_label = QLabel("Select Mode")
        self.mode_label.setFont(bfont)
        self.mode_group = QButtonGroup()
        self.dual_mode_radio = QRadioButton("Dual Channel")
        self.single_mode_radio = QRadioButton("Single Channel")
        self.dual_mode_radio.setChecked(True)
        self.mode_group.addButton(self.dual_mode_radio, 0)
        self.mode_group.addButton(self.single_mode_radio, 1)
        self.dual_mode_radio.toggled.connect(self.on_mode_changed)

        self.enhance_button = QPushButton("Reconstruct")
        self.enhance_button.setFixedSize(120, 50)
        self.enhance_button.clicked.connect(self.enhance_videos)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setFixedSize(120, 50)
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_reconstruction)
        self.update_button_styles()

        # 进度条
        self.enhance_progress = QProgressBar()
        self.enhance_progress.setRange(0, 100)

        # System monitoring widgets
        self.monitor_label = QLabel("System Monitor")
        self.monitor_label.setFont(bfont)
        self.cpu_monitor_label = QLabel("CPU: -- %  |  Memory: -- %")
        self.gpu_monitor_label = QLabel("GPU: Not monitoring")
        self.gpu_monitor_label.setWordWrap(True)
        
        # Create a container for monitoring info
        self.monitor_container = QWidget()
        monitor_layout = QVBoxLayout()
        monitor_layout.addWidget(self.monitor_label)
        monitor_layout.addWidget(self.cpu_monitor_label)
        monitor_layout.addWidget(self.gpu_monitor_label)
        monitor_layout.setContentsMargins(5, 5, 5, 5)
        self.monitor_container.setLayout(monitor_layout)
        self.monitor_container.setStyleSheet("QWidget { background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 5px; }")
        # System Monitor 一直显示，不再隐藏

    def create_layout(self):
        central_widget = QWidget()
        layout = QHBoxLayout()
        in_layout = QVBoxLayout()
        out_layout = QVBoxLayout()
        setting_layout = QVBoxLayout()
        gain_layout = QHBoxLayout()
        gain_value_layout = QVBoxLayout()
        model_layout = QHBoxLayout()
        model_path_layout = QVBoxLayout()
        opt_path_layout = QVBoxLayout()
        output_path_layout = QVBoxLayout()
        select_channel_layout = QHBoxLayout()

        # Dual channel layout (default)
        in_layout.addWidget(self.input_label1)
        input_path1_layout = QHBoxLayout()
        input_path1_layout.addWidget(self.input_path1)
        input_path1_layout.addWidget(self.input_browse1)
        in_layout.addLayout(input_path1_layout)
        in_layout.addWidget(self.input_video1)
        in_layout.addWidget(self.in1_sld)
        in_layout.addWidget(self.output_label1)
        in_layout.addWidget(self.output_video1)
        in_layout.addWidget(self.out1_sld)

        out_layout.addWidget(self.input_label2)
        input_path2_layout = QHBoxLayout()
        input_path2_layout.addWidget(self.input_path2)
        input_path2_layout.addWidget(self.input_browse2)
        out_layout.addLayout(input_path2_layout)
        out_layout.addWidget(self.input_video2)
        out_layout.addWidget(self.in2_sld)
        out_layout.addWidget(self.output_label2)
        out_layout.addWidget(self.output_video2)
        out_layout.addWidget(self.out2_sld)
        
        # Single channel layout (initially hidden)
        single_layout = QVBoxLayout()
        single_layout.addWidget(self.single_input_label)
        single_input_path_layout = QHBoxLayout()
        single_input_path_layout.addWidget(self.single_input_path)
        single_input_path_layout.addWidget(self.single_input_browse)
        single_layout.addLayout(single_input_path_layout)
        single_layout.addWidget(self.single_input_video)
        single_layout.addWidget(self.single_in_sld)
        single_layout.addWidget(self.single_output_label)
        single_layout.addWidget(self.single_output_video)
        single_layout.addWidget(self.single_out_sld)
        
        # Create container widgets for layouts
        in_widget = QWidget()
        in_widget.setLayout(in_layout)
        out_widget = QWidget()
        out_widget.setLayout(out_layout)
        self.single_layout_widget = QWidget()
        self.single_layout_widget.setLayout(single_layout)
        self.single_layout_widget.hide()
        
        # Store references for show/hide
        self.in_widget = in_widget
        self.out_widget = out_widget
        
        layout.addWidget(in_widget)
        layout.addWidget(out_widget)
        layout.addWidget(self.single_layout_widget)

        setting_layout.addWidget(self.instructions_label)
        # 增加分割线
        setting_layout.addWidget(QLabel("----------------------------------------------------------------------------"))
        gain_value_layout.addWidget(self.linear_gain_label)
        gain_value_layout.addWidget(self.linear_gain_input)
        gain_value_layout.addWidget(self.linear_gain_set_button)
        gain_layout.addLayout(gain_value_layout)
        setting_layout.addLayout(gain_layout)
        # 增加分割线
        setting_layout.addWidget(QLabel("----------------------------------------------------------------------------"))

        model_path_layout.addWidget(self.model_path_label)
        model_path_layout.addWidget(self.model_path_show)
        model_path_layout.addWidget(self.browse_model_button)
        
        opt_path_layout.addWidget(self.opt_path_label)
        opt_path_layout.addWidget(self.opt_path_show)
        opt_path_layout.addWidget(self.browse_opt_button)

        output_path_layout.addWidget(self.output_label)
        output_path_layout.addWidget(self.output_path_show)
        output_path_layout.addWidget(self.browse_output_button)

        model_layout.addLayout(model_path_layout)
        model_layout.addLayout(opt_path_layout)
        model_layout.addLayout(output_path_layout)
        setting_layout.addLayout(model_layout)

        # 增加分割线
        setting_layout.addWidget(QLabel("----------------------------------------------------------------------------"))

        # Mode selection
        setting_layout.addWidget(self.mode_label)
        mode_radio_layout = QHBoxLayout()
        mode_radio_layout.addWidget(self.dual_mode_radio)
        mode_radio_layout.addWidget(self.single_mode_radio)
        setting_layout.addLayout(mode_radio_layout)
        # 增加分割线
        setting_layout.addWidget(QLabel("----------------------------------------------------------------------------"))
        setting_layout.addWidget(self.device_label)
        setting_layout.addWidget(self.device_tree)
        # 增加分割线
        setting_layout.addWidget(QLabel("----------------------------------------------------------------------------"))
        setting_layout.addWidget(self.enhance_progress)
        
        # 创建水平布局，让 System Monitor 和 Reconstruct 按钮并排显示
        monitor_button_layout = QHBoxLayout()
        monitor_button_layout.addWidget(self.monitor_container)  # System Monitor 占据剩余空间
        button_layout = QVBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(0)
        button_layout.addWidget(self.enhance_button)
        button_layout.addWidget(self.stop_button)
        button_widget = QWidget()
        button_widget.setFixedSize(120, 100)
        button_widget.setLayout(button_layout)
        monitor_button_layout.addWidget(button_widget)  # Buttons 在右侧
        monitor_button_layout.setStretch(0, 1)  # Monitor 可以拉伸
        monitor_button_layout.setStretch(1, 0)  # Button 固定宽度
        
        # 创建容器来包含这个水平布局
        monitor_button_widget = QWidget()
        monitor_button_widget.setLayout(monitor_button_layout)
        setting_layout.addWidget(monitor_button_widget)
        
        # 创建设置面板容器并设置最小宽度
        setting_widget = QWidget()
        setting_widget.setLayout(setting_layout)
        setting_widget.setMinimumWidth(480)  # 设置右侧面板最小宽度，确保文字显示完整
        layout.addWidget(setting_widget)

        central_widget.setLayout(layout)
        # 设置不允许窗口拉伸 - 增加窗口宽度以显示完整的右侧面板
        self.setFixedSize(1400, 750)
        self.setCentralWidget(central_widget)

    def pad_to_square(self, show_image):
        max_h_w = max(show_image.shape)
        padded_image = np.zeros((max_h_w, max_h_w), dtype=np.uint8)
        pad_h, pad_w = (max_h_w - show_image.shape[0]) // 2, (max_h_w - show_image.shape[1]) // 2
        padded_image[pad_h : pad_h + show_image.shape[0], pad_w : pad_w + show_image.shape[1]] = show_image
        return padded_image

    def on_mode_changed(self):
        """Handle mode change between single and dual channel"""
        if self.dual_mode_radio.isChecked():
            self.mode = "dual"
            # Show dual channel widgets, hide single channel widgets
            self.in_widget.show()
            self.out_widget.show()
            self.single_layout_widget.hide()
        else:
            self.mode = "single"
            # Hide dual channel widgets, show single channel widgets
            self.in_widget.hide()
            self.out_widget.hide()
            self.single_layout_widget.show()
        print(f"Mode changed to: {self.mode}")

    def changeValueSingleIn(self, value):
        """Handle single channel input slider change"""
        if self.single_ch_numpy is not None and value >= 0:
            if value < self.single_ch_numpy.shape[0]:
                show_image = np.clip(self.single_ch_numpy[value] * self.linear_gain * 255, 0, 255).astype(np.uint8)
                show_image = self.pad_to_square(show_image)
                h, w = show_image.shape
                self.single_input_video.setPixmap(QPixmap.fromImage(QImage(show_image, w, h, w, QImage.Format.Format_Grayscale8)))

    def changeValueSingleOut(self, value):
        """Handle single channel output slider change"""
        if self.single_ch_out_numpy is not None and value >= 0:
            if value < self.single_ch_out_numpy.shape[0]:
                show_image = np.clip(self.single_ch_out_numpy[value] * 255, 0, 255).astype(np.uint8)
                show_image = self.pad_to_square(show_image)
                h, w = show_image.shape
                self.single_output_video.setPixmap(QPixmap.fromImage(QImage(show_image, w, h, w, QImage.Format.Format_Grayscale8)))

    def changeChnAChecked(self, state):
        self.enhance_a = state == Qt.CheckState.Checked
        if not self.enhance_a and not self.enhance_b:
            self.enhance_b = True
            self.ch_b_box.setChecked(True)
        self.changeValueIn1(self.in1_sld.value())
        print(f"Enhance A: {self.enhance_a}, Enhance B: {self.enhance_b}")

    def changeChnBChecked(self, state):
        self.enhance_b = state == Qt.CheckState.Checked
        if not self.enhance_a and not self.enhance_b:
            self.enhance_a = True
            self.ch_a_box.setChecked(True)
        self.changeValueIn2(self.in2_sld.value())
        print(f"Enhance A: {self.enhance_a}, Enhance B: {self.enhance_b}")

    def changeValueIn1(self, value):
        # print(f"Slider value: {value}")
        if self.ch_a_numpy is not None and value >= 0:
            if value < self.ch_a_numpy.shape[0]:
                show_image = np.clip(self.ch_a_numpy[value] * self.linear_gain * 255, 0, 255).astype(np.uint8)
                show_image = self.pad_to_square(show_image) * self.enhance_a
                h, w = show_image.shape
                self.input_video1.setPixmap(QPixmap.fromImage(QImage(show_image, w, h, w, QImage.Format.Format_Grayscale8)))

    def changeValueIn2(self, value):
        # print(f"Slider value: {value}")
        if self.ch_b_numpy is not None and value >= 0:
            if value < self.ch_b_numpy.shape[0]:
                show_image = np.clip(self.ch_b_numpy[value] * self.linear_gain * 255, 0, 255).astype(np.uint8)
                show_image = self.pad_to_square(show_image) * self.enhance_b
                h, w = show_image.shape
                self.input_video2.setPixmap(QPixmap.fromImage(QImage(show_image, w, h, w, QImage.Format.Format_Grayscale8)))

    def changeValueOut1(self, value):
        # print(f"Slider value: {value}")
        if self.ch_a_out_numpy is not None and value >= 0:
            if value < self.ch_a_out_numpy.shape[0]:
                show_image = np.clip(self.ch_a_out_numpy[value] * 255, 0, 255).astype(np.uint8)
                show_image = self.pad_to_square(show_image) * self.enhance_a
                h, w = show_image.shape
                self.output_video1.setPixmap(QPixmap.fromImage(QImage(show_image, w, h, w, QImage.Format.Format_Grayscale8)))

    def changeValueOut2(self, value):
        # print(f"Slider value: {value}")
        if self.ch_b_out_numpy is not None and value >= 0:
            if value < self.ch_b_numpy.shape[0]:
                show_image = np.clip(self.ch_b_out_numpy[value] * 255, 0, 255).astype(np.uint8)
                show_image = self.pad_to_square(show_image) * self.enhance_b
                h, w = show_image.shape
                self.output_video2.setPixmap(QPixmap.fromImage(QImage(show_image, w, h, w, QImage.Format.Format_Grayscale8)))

    def set_linear_gain(self):
        if self.mode == "dual":
            self.changeValueIn1(self.in1_sld.value())
            self.changeValueIn2(self.in2_sld.value())
        else:
            self.changeValueSingleIn(self.single_in_sld.value())

    def resolve_input_paths(self, file_path):
        if os.path.isdir(file_path):
            matches = []
            for root, _, files in os.walk(file_path):
                for fname in files:
                    if fname.lower().endswith((".tif", ".tiff")):
                        abs_path = opj(root, fname)
                        rel_path = os.path.relpath(abs_path, file_path)
                        matches.append((abs_path, rel_path))
            matches.sort(key=lambda x: x[1])
            return matches
        if os.path.isfile(file_path):
            return [(file_path, os.path.basename(file_path))]
        return []

    def set_input_path_from_edit(self, name):
        if name == "single":
            file_path = self.single_input_path.text().strip()
        elif name == "a":
            file_path = self.input_path1.text().strip()
        else:
            file_path = self.input_path2.text().strip()
        if not file_path:
            return
        self.loadAndSetInputPath(file_path, name)

    def build_input_browse_menu(self, name):
        menu = QMenu(self)
        action_file = menu.addAction("Select File")
        action_folder = menu.addAction("Select Folder")
        action_file.triggered.connect(lambda: self.browse_input_path(name, "file"))
        action_folder.triggered.connect(lambda: self.browse_input_path(name, "folder"))
        return menu

    def browse_input_path(self, name, mode="file"):
        try:
            if mode == "folder":
                path = self.open_folder_dialog("Select Input Folder")
            else:
                path = self.open_file_dialog(
                    "Select Input TIFF",
                    "TIFF Files (*.tif *.tiff);;All Files (*)",
                    ".",
                )
            if path:
                self.loadAndSetInputPath(path, name)
        except Exception:
            print(traceback.format_exc())

    def loadAndSetInputPath(self, file_path, name="a"):
        if name == "single":
            in_slider = self.single_in_sld
            out_slider = self.single_out_sld
            path_label = self.single_input_path
            video_viewer = self.single_input_video
        elif name == "a":
            in_slider = self.in1_sld
            out_slider = self.out1_sld
            path_label = self.input_path1
            video_viewer = self.input_video1
        else:  # name == "b"
            in_slider = self.in2_sld
            out_slider = self.out2_sld
            path_label = self.input_path2
            video_viewer = self.input_video2

        input_paths = self.resolve_input_paths(file_path)
        if not input_paths:
            msg_box = QMessageBox()
            msg_box.setWindowTitle("Load input error!")
            msg_box.setText(f"Input path does not exist or contains no TIFF files: {file_path}")
            msg_box.setIcon(QMessageBox.Icon.Critical)
            msg_box.addButton(QMessageBox.StandardButton.Ok)
            msg_box.exec()
            return None

        abs_paths = [p for p, _ in input_paths]
        rel_paths = [r for _, r in input_paths]
        is_dir = os.path.isdir(file_path)

        if name == "single":
            self.single_paths = abs_paths
            self.single_rel_paths = rel_paths
            self.single_is_dir = is_dir
        elif name == "a":
            self.ch_a_paths = abs_paths
            self.ch_a_rel_paths = rel_paths
            self.ch_a_is_dir = is_dir
        else:
            self.ch_b_paths = abs_paths
            self.ch_b_rel_paths = rel_paths
            self.ch_b_is_dir = is_dir

        path_label.setText(file_path)
        if len(abs_paths) > 1:
            path_label.setToolTip(f"Batch mode: {len(abs_paths)} files")
        else:
            path_label.setToolTip(file_path)

        data = load_tiff_seq(abs_paths[0])
        print(f"numpy.shape: {data.shape}")
        in_slider.setMaximum(data.shape[0] - 1)
        in_slider.setMinimum(0)
        out_slider.setMaximum(data.shape[0] - 1)
        out_slider.setMinimum(0)

        show_image = np.clip(data[0] * self.linear_gain * 255, 0, 255).astype(np.uint8)
        show_image = self.pad_to_square(show_image)
        h, w = show_image.shape
        video_viewer.setPixmap(QPixmap.fromImage(QImage(show_image, w, h, w, QImage.Format.Format_Grayscale8)))
        return data

    def eventFilter(self, obj, event):
        if hasattr(self, "input_video1") and hasattr(self, "input_video2") and hasattr(self, "single_input_video"):
            if obj == self.input_video1 or obj == self.input_video2 or obj == self.single_input_video:
                if event.type() == QEvent.Type.DragEnter:
                    if event.mimeData().hasUrls():
                        event.acceptProposedAction()
                elif event.type() == QEvent.Type.Drop:
                    urls = event.mimeData().urls()
                    if urls:
                        file_path = urls[0].toLocalFile()
                        if obj == self.input_video1:
                            self.ch_a_numpy = self.loadAndSetInputPath(file_path, "a")
                        elif obj == self.input_video2:
                            self.ch_b_numpy = self.loadAndSetInputPath(file_path, "b")
                        elif obj == self.single_input_video:
                            self.single_ch_numpy = self.loadAndSetInputPath(file_path, "single")
        
        # 处理 model_path_show 的拖放
        if hasattr(self, "model_path_show") and obj == self.model_path_show:
            if event.type() == QEvent.Type.DragEnter:
                if event.mimeData().hasUrls():
                    event.acceptProposedAction()
            elif event.type() == QEvent.Type.Drop:
                urls = event.mimeData().urls()
                if urls:
                    file_path = urls[0].toLocalFile()
                    # 检查是否是模型文件 (.pth 或 .pkl)
                    if file_path.endswith(('.pth', '.pkl')):
                        if os.path.exists(file_path):
                            self.model_path = file_path
                            self.model_path_show.setPlainText("Path: " + file_path)
                            # 尝试自动检测对应的 yaml 文件
                            base_path = file_path[: file_path.rfind(".")]
                            suggested_yaml = base_path + ".yaml"
                            if os.path.exists(suggested_yaml):
                                self.opt_path = suggested_yaml
                                self.opt_path_show.setPlainText("Path: " + suggested_yaml)
                                print(f"Auto-detected YAML config: {suggested_yaml}")
        
        return super().eventFilter(obj, event)

    def browse_output_folder(self):
        folder_path = self.open_folder_dialog("Select Output Folder")
        if folder_path:
            self.output_folder = folder_path
            self.output_path_show.setPlainText("Path: " + folder_path)
            print(f"Output folder: {self.output_folder}")

    def browse_model(self):
        """Browse and select model file (.pth or .pkl)"""
        try:
            path = self.open_file_dialog(
                "Select Network Model",
                "Model Files (*.pth *.pkl);;PTH Files (*.pth);;PKL Files (*.pkl);;All Files (*)",
                "./GUI_Utils/Model",
            )
            if path:
                self.model_path = path
                self.model_path_show.setPlainText("Path: " + path)
                
                # Try to auto-suggest yaml file (same name in same directory)
                base_path = path[: path.rfind(".")]
                suggested_yaml = base_path + ".yaml"
                if os.path.exists(suggested_yaml):
                    # Auto-update yaml path if found
                    self.opt_path = suggested_yaml
                    self.opt_path_show.setPlainText("Path: " + suggested_yaml)
                    print(f"Auto-detected YAML config: {suggested_yaml}")
                else:
                    print(f"Model file selected: {path}")
                    print(f"YAML file not auto-detected. Please select config file manually if needed.")
        except Exception as e:
            print(traceback.format_exc())
    
    def browse_config(self):
        """Browse and select YAML configuration file"""
        try:
            path = self.open_file_dialog(
                "Select Configuration File (YAML)",
                "YAML Files (*.yaml *.yml);;All Files (*)",
                "./GUI_Utils/Model",
            )
            if path:
                self.opt_path = path
                self.opt_path_show.setPlainText("Path: " + path)
                print(f"Config file selected: {path}")
        except Exception as e:
            print(traceback.format_exc())

    def open_file_dialog(self, title, file_filter, base_dir="."):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            title,
            base_dir,
            file_filter,
        )
        return file_path

    def open_folder_dialog(self, title, base_dir="."):
        return QFileDialog.getExistingDirectory(
            self,
            title,
            base_dir,
            QFileDialog.Option.ShowDirsOnly,
        )

    def device_selected(self, index):
        print(f"Selected device: {self.device_model.data(index)}")
        self.selected_device = self.device_model.data(index)
        self.selected_device = self.selected_device[self.selected_device.find("(") + 1 : self.selected_device.find(")")]

    def enhance_videos(self):
        if self.reconstruct_thread is not None and self.reconstruct_thread.isRunning():
            msg_box = QMessageBox()
            msg_box.setWindowTitle("Warning!")
            msg_box.setText("Reconstruction is already running. Please stop it before starting a new one.")
            msg_box.setIcon(QMessageBox.Icon.Information)
            msg_box.addButton(QMessageBox.StandardButton.Ok)
            msg_box.exec()
            return
        if self.selected_device == "cpu":
            # 弹出提示框
            msg_box = QMessageBox()
            msg_box.setWindowTitle("Warning!")
            msg_box.setText(f"Running on CPU will be extremely slow. \nWe recommend using GPU to run DAWN. ")
            msg_box.setIcon(QMessageBox.Icon.Information)
            msg_box.addButton(QMessageBox.StandardButton.Ok)
            msg_box.exec()

        # 调用另一个线程处理
        input_paths_a = None
        input_paths_b = None
        input_paths_single = None
        if self.mode == "single":
            if not self.single_paths and self.single_ch_numpy is None:
                msg_box = QMessageBox()
                msg_box.setWindowTitle("Error!")
                msg_box.setText("Please load single channel input video first.")
                msg_box.exec()
                return
            print(f"Mode: Single, Linear Gain: {self.linear_gain}, Device: {self.selected_device}")
            input_paths_single = self.single_paths if self.single_paths else None
            ch_a_numpy = None
            ch_b_numpy = None
            single_ch_numpy = None if input_paths_single else (self.single_ch_numpy * self.linear_gain)
        else:
            if (not self.ch_a_paths or not self.ch_b_paths) and (self.ch_a_numpy is None or self.ch_b_numpy is None):
                msg_box = QMessageBox()
                msg_box.setWindowTitle("Error!")
                msg_box.setText("Please load both Acceptor and Donor channel videos first.")
                msg_box.exec()
                return
            if self.ch_a_paths and self.ch_b_paths and (self.ch_a_is_dir != self.ch_b_is_dir):
                msg_box = QMessageBox()
                msg_box.setWindowTitle("Error!")
                msg_box.setText("Acceptor and Donor must both be files or both be folders for batch mode.")
                msg_box.exec()
                return
            if self.ch_a_paths and self.ch_b_paths and self.ch_a_is_dir and self.ch_b_is_dir:
                map_b = {rel: path for path, rel in zip(self.ch_b_paths, self.ch_b_rel_paths)}
                paired_a = []
                paired_b = []
                missing = []
                for path_a, rel in zip(self.ch_a_paths, self.ch_a_rel_paths):
                    if rel in map_b:
                        paired_a.append(path_a)
                        paired_b.append(map_b[rel])
                    else:
                        missing.append(rel)
                if missing:
                    msg_box = QMessageBox()
                    msg_box.setWindowTitle("Error!")
                    msg_box.setText(f"Donor folder is missing {len(missing)} file(s). Example: {missing[0]}")
                    msg_box.exec()
                    return
                self.ch_a_paths = paired_a
                self.ch_b_paths = paired_b
            elif self.ch_a_paths and self.ch_b_paths and len(self.ch_a_paths) != len(self.ch_b_paths):
                msg_box = QMessageBox()
                msg_box.setWindowTitle("Error!")
                msg_box.setText("Acceptor and Donor input folders must contain the same number of TIFF files.")
                msg_box.exec()
                return
            print(f"Mode: Dual, Linear Gain: {self.linear_gain}, Device: {self.selected_device}, Enhance channel: A {self.enhance_a}, B {self.enhance_b}")
            input_paths_a = self.ch_a_paths if self.ch_a_paths else None
            input_paths_b = self.ch_b_paths if self.ch_b_paths else None
            ch_a_numpy = None if input_paths_a else (self.ch_a_numpy * self.linear_gain * self.enhance_a)
            ch_b_numpy = None if input_paths_b else (self.ch_b_numpy * self.linear_gain * self.enhance_b)
            single_ch_numpy = None
        
        t = ReconstructThread(
            self,
            self.opt_path,
            self.model_path,
            self.mode,
            ch_a_numpy,
            ch_b_numpy,
            single_ch_numpy,
            self.selected_device,
            self.enhance_progress,
            self.output_folder,
            self.linear_gain,
            input_paths_a=input_paths_a,
            input_paths_b=input_paths_b,
            input_paths_single=input_paths_single,
            enhance_a=self.enhance_a,
            enhance_b=self.enhance_b,
        )
        t.resultReady.connect(self.enhance_finished)
        t.updateProgress.connect(self.enhance_progress.setValue)
        t.error.connect(self.enhance_error)
        t.canceled.connect(self.enhance_canceled)
        
        # Start system monitoring
        self.start_monitoring()
        
        t.start()
        self.reconstruct_thread = t
        self.enhance_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.update_button_styles()
        # t.quit()

    def enhance_finished(self, result):
        # print("Reconstruct finished: result", result)
        self.enhance_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.reconstruct_thread = None
        self.update_button_styles()
        
        # Stop system monitoring
        self.stop_monitoring()
        
        ch_a_out, ch_b_out, result_dir = result
        
        if self.mode == "single":
            self.single_ch_out_numpy = ch_a_out
            self.changeValueSingleOut(self.single_out_sld.value())
        else:
            self.ch_a_out_numpy = ch_a_out
            self.ch_b_out_numpy = ch_b_out
            self.ch_a_out_numpy *= self.enhance_a
            self.ch_b_out_numpy *= self.enhance_b
            self.changeValueOut1(self.out1_sld.value())
            self.changeValueOut2(self.out2_sld.value())
        
        # 弹出提示框
        msg_box = QMessageBox()
        msg_box.setWindowTitle("Success!")
        msg_box.setText(f"Video Reconstruction Finished! \nReconstructed videos are saved to {os.path.abspath(result_dir)}")
        msg_box.setIcon(QMessageBox.Icon.Information)
        open_folder_button = QPushButton("  Open folder  ")
        msg_box.addButton(open_folder_button, QMessageBox.ButtonRole.ActionRole)
        msg_box.addButton(QMessageBox.StandardButton.Ok)
        # 转换为绝对路径
        output_folder_abs = os.path.abspath(result_dir)
        open_folder_button.clicked.connect(lambda: subprocess.Popen(f'explorer /select, "{output_folder_abs}"'))
        msg_box.exec()
        self.enhance_progress.setValue(0)

    def enhance_error(self, result):
        # 弹出提示框
        msg_box = QMessageBox()
        msg_box.setWindowTitle("Error!")
        msg_box.setText(f"Video Reconstruction Error! Please check the selected model and device. \n{str(result)}")
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.addButton(QMessageBox.StandardButton.Ok)
        msg_box.exec()
        self.enhance_progress.setValue(0)
        self.enhance_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.reconstruct_thread = None
        self.update_button_styles()
        self.stop_monitoring()

    def enhance_canceled(self, result):
        msg_box = QMessageBox()
        msg_box.setWindowTitle("Canceled")
        msg_box.setText(f"Reconstruction canceled.\n{str(result)}")
        msg_box.setIcon(QMessageBox.Icon.Information)
        msg_box.addButton(QMessageBox.StandardButton.Ok)
        msg_box.exec()
        self.enhance_progress.setValue(0)
        self.enhance_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.reconstruct_thread = None
        self.update_button_styles()
        self.stop_monitoring()

    def stop_reconstruction(self):
        if self.reconstruct_thread is not None and self.reconstruct_thread.isRunning():
            self.reconstruct_thread.request_stop()
            self.stop_button.setEnabled(False)
            self.update_button_styles()

    def update_button_styles(self):
        self.enhance_button.setStyleSheet(
            "background-color: #000000; color: white;" if self.enhance_button.isEnabled() else "background-color: #C0C0C0; color: #555555;"
        )
        self.stop_button.setStyleSheet(
            "background-color: #000000; color: white;" if self.stop_button.isEnabled() else "background-color: #C0C0C0; color: #555555;"
        )
    
    def start_monitoring(self):
        """Start system monitoring"""
        if self.monitor_thread is None or not self.monitor_thread.isRunning():
            # monitor_container 已经一直显示，不需要再 show()
            self.monitor_thread = MonitorThread(self, interval=0.5)  # Update every 0.5 seconds
            self.monitor_thread.metricsUpdated.connect(self.update_monitor_display)
            self.monitor_thread.start()
            self.monitoring_active = True
            print("System monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        if self.monitor_thread is not None and self.monitor_thread.isRunning():
            self.monitor_thread.stop()
            self.monitor_thread = None
            self.monitoring_active = False
            # monitor_container 一直显示，不再隐藏
            print("System monitoring stopped")
    
    def update_monitor_display(self, metrics):
        """Update the monitoring display with new metrics"""
        try:
            # Update CPU info
            cpu_text = f"CPU: {metrics.cpu_percent:.1f}%  |  Memory: {metrics.cpu_memory_percent:.1f}% ({metrics.cpu_memory_used_gb:.1f}/{metrics.cpu_memory_total_gb:.1f} GB)"
            self.cpu_monitor_label.setText(cpu_text)
            
            # Update GPU info
            if metrics.gpu_count > 0 and len(metrics.gpu_metrics) > 0:
                gpu_lines = []
                for gpu in metrics.gpu_metrics:
                    gpu_line = (f"GPU {gpu['device_id']} ({gpu['device_name']}): "
                               f"Util {gpu['gpu_util']}%  |  "
                               f"Memory: {gpu['memory_percent']:.1f}% ({gpu['memory_used_gb']:.1f}/{gpu['memory_total_gb']:.1f} GB)")
                    if gpu['temperature'] > 0:
                        gpu_line += f"  |  Temp: {gpu['temperature']}°C"
                    if gpu['power_usage'] > 0:
                        gpu_line += f"  |  Power: {gpu['power_usage']:.1f}W"
                    gpu_lines.append(gpu_line)
                self.gpu_monitor_label.setText("\n".join(gpu_lines))
            else:
                self.gpu_monitor_label.setText("GPU: Not available")
        except Exception as e:
            print(f"Error updating monitor display: {e}")
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Stop monitoring thread if running
        if self.monitor_thread is not None and self.monitor_thread.isRunning():
            self.stop_monitoring()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoReconstructionApp()
    window.show()
    sys.exit(app.exec())
