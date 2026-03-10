# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['Train_GUI.py'],
    pathex=[],
    binaries=[('C:\\Users\\Admin\\.conda\\envs\\DAWN_code\\DLLs\\_ctypes.pyd', '.'), ('C:\\Users\\Admin\\.conda\\envs\\DAWN_code\\Library\\bin\\ffi.dll', '.'), ('C:\\Users\\Admin\\.conda\\envs\\DAWN_code\\Library\\bin\\ffi-7.dll', '.'), ('C:\\Users\\Admin\\.conda\\envs\\DAWN_code\\Library\\bin\\ffi-8.dll', '.')],
    datas=[('GUI_Utils', 'GUI_Utils'), ('config_templates', 'config_templates')],
    hiddenimports=['PyQt6.QtCore', 'PyQt6.QtGui', 'PyQt6.QtWidgets', 'ctypes', '_ctypes', 'matplotlib', 'matplotlib.backends.backend_qtagg', 'yaml', 'omegaconf', 'psutil', 'pynvml', 'numpy'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='DAWN_TrainGUI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['GUI_Utils\\icon.png'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='DAWN_TrainGUI',
)
