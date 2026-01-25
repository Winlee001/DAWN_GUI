# -*- mode: python ; coding: utf-8 -*-
# PyInstaller 配置文件 - 用于打包 DAWN GUI 应用程序（目录模式，避免路径长度限制）

import sys
import os

block_cipher = None

# 收集所有需要的数据文件
datas = [
    ('GUI_Utils', 'GUI_Utils'),  # 包含整个 GUI_Utils 目录（图标、模型等）
]

# 隐藏导入列表 - PyInstaller 可能无法自动检测的模块
hiddenimports = [
    # PyQt6 相关
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.QtWidgets',
    'PyQt6.QtCore.QUrl',
    'PyQt6.QtCore.QThread',
    'PyQt6.QtCore.pyqtSignal',
    
    # PyTorch 相关
    'torch',
    'torch.nn',
    'torch.nn.functional',
    'torch.utils.data',
    'torch.utils.data.dataloader',
    'torch.cuda',
    'torchvision',
    
    # 科学计算库
    'numpy',
    'numpy.core._methods',
    'numpy.lib.format',
    'scipy',
    'scipy.sparse',
    'scipy.sparse.csgraph',
    
    # 图像处理
    'tifffile',
    'cv2',
    'PIL',
    'PIL.Image',
    'PIL.ImageTk',
    'skimage',
    'skimage.io',
    'skimage.measure',
    'imageio',
    'imageio.plugins.pillow',
    
    # 配置管理
    'omegaconf',
    'yaml',
    'yaml.loader',
    
    # 系统工具
    'psutil',
    'pynvml',  # GPU 监控（可选）
    
    # 其他工具
    'tqdm',
    'packaging',
    
    # 自定义模块
    'GUI_Utils.Utils',
    'GUI_Utils.SystemMonitor',
    'DAWN_Inference',
    'Test',
    'Net.DBSN',
    'Net.DCMAN',
    'Net.SCMAN',
    'Net.Utils',
    'Net.DAWN_net',
    'Net.DAWN_net.backbone_net',
    'Net.DAWN_net.DBSN_fusion',
    'Net.DAWN_net.DBSN_centrblind',
    'Net.LSTM',
    'Net.LSTM.BiConvLSTM',
    'Net.DualDecoder',
    'TrainUtils.TrainUtils',
    'TrainUtils.MinibatchGenerater',
    'TrainUtils.OptType',
    'ImgProcessing.ImgUtils',
]

# 分析主程序
a = Analysis(
    ['DAWN_GUI.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # 排除不需要的模块以减小文件大小
        'tensorboard',
        'tensorboardX',
        'jupyter',
        'notebook',
        'IPython',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# 创建 PYZ 归档
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# 创建可执行文件（目录模式）
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,  # 关键：设置为 True，使用目录模式
    name='DAWN_GUI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # 保留控制台窗口（用于调试），改为 False 可隐藏
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='GUI_Utils/icon.png',  # 应用程序图标
)

# 收集所有文件到目录（目录模式）
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='DAWN_GUI',
)