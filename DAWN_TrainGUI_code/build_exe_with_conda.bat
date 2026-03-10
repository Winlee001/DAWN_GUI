@echo off
chcp 65001 >nul
echo ========================================
echo DAWN Train GUI 打包脚本（Conda 环境版本）
echo ========================================
echo.

REM 设置 conda 环境路径
set CONDA_ENV=C:\Users\Admin\.conda\envs\DAWN_code
set CONDA_BASE=C:\Users\Admin\.conda

REM 检查 conda 环境是否存在
if not exist "%CONDA_ENV%" (
    echo [错误] Conda 环境不存在: %CONDA_ENV%
    echo 请检查路径是否正确
    pause
    exit /b 1
)

echo [信息] 检测到 Conda 环境: %CONDA_ENV%
echo.

REM 切换到脚本所在目录
cd /d "%~dp0"
echo [信息] 当前工作目录: %CD%
echo.

REM 初始化 conda（如果需要）
if exist "%CONDA_BASE%\Scripts\activate.bat" (
    call "%CONDA_BASE%\Scripts\activate.bat"
) else if exist "%CONDA_BASE%\etc\profile.d\conda.sh" (
    call "%CONDA_BASE%\etc\profile.d\conda.sh"
) else (
    echo [警告] 无法找到 conda 初始化脚本，尝试直接使用环境中的 Python
)

REM 激活 conda 环境
echo [信息] 激活 Conda 环境: DAWN_code
call conda activate DAWN_code
if errorlevel 1 (
    echo [错误] 无法激活 conda 环境
    echo 尝试直接使用环境中的 Python...
    set PYTHON_EXE=%CONDA_ENV%\python.exe
) else (
    set PYTHON_EXE=python
)

REM 验证 Python 路径
if /I "%PYTHON_EXE%"=="python" (
    where python >nul 2>nul
    if errorlevel 1 (
        if not exist "%CONDA_ENV%\python.exe" (
            echo [错误] 无法找到 Python 解释器
            pause
            exit /b 1
        ) else (
            set PYTHON_EXE=%CONDA_ENV%\python.exe
        )
    )
) else (
    if not exist "%PYTHON_EXE%" (
        if not exist "%CONDA_ENV%\python.exe" (
            echo [错误] 无法找到 Python 解释器
            pause
            exit /b 1
        ) else (
            set PYTHON_EXE=%CONDA_ENV%\python.exe
        )
    )
)

echo [信息] 使用 Python: %PYTHON_EXE%
echo [信息] Python 版本:
"%PYTHON_EXE%" --version
echo.

REM 检查 PyInstaller 是否安装
echo [信息] 检查 PyInstaller...
"%PYTHON_EXE%" -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo [信息] PyInstaller 未安装，正在安装...
    "%PYTHON_EXE%" -m pip install pyinstaller
    if errorlevel 1 (
        echo [错误] PyInstaller 安装失败！
        pause
        exit /b 1
    )
) else (
    echo [信息] PyInstaller 已安装
)
echo.

REM 清理之前的构建
if exist build (
    echo [信息] 清理 build 目录...
    rmdir /s /q build
)
if exist dist (
    echo [信息] 清理 dist 目录...
    rmdir /s /q dist
)

echo [信息] 开始打包 DAWN Train GUI（这可能需要几分钟）...
echo.
"%PYTHON_EXE%" -m PyInstaller ^
    --name=DAWN_TrainGUI ^
    --icon=GUI_Utils/icon.png ^
    --add-data "GUI_Utils;GUI_Utils" ^
    --add-data "config_templates;config_templates" ^
    --add-binary "%CONDA_ENV%\DLLs\_ctypes.pyd;." ^
    --add-binary "%CONDA_ENV%\Library\bin\ffi.dll;." ^
    --add-binary "%CONDA_ENV%\Library\bin\ffi-7.dll;." ^
    --add-binary "%CONDA_ENV%\Library\bin\ffi-8.dll;." ^
    --hidden-import=PyQt6.QtCore ^
    --hidden-import=PyQt6.QtGui ^
    --hidden-import=PyQt6.QtWidgets ^
    --hidden-import=ctypes ^
    --hidden-import=_ctypes ^
    --hidden-import=matplotlib ^
    --hidden-import=matplotlib.backends.backend_qtagg ^
    --hidden-import=yaml ^
    --hidden-import=omegaconf ^
    --hidden-import=psutil ^
    --hidden-import=pynvml ^
    --hidden-import=numpy ^
    --windowed ^
    Train_GUI.py

if errorlevel 1 (
    echo.
    echo [错误] 打包失败！请查看上面的错误信息
    pause
    exit /b 1
)

REM 检查打包结果
if not exist "dist\DAWN_TrainGUI\DAWN_TrainGUI.exe" (
    echo.
    echo [错误] 未找到生成的可执行文件！
    pause
    exit /b 1
)

echo.
echo ========================================
echo [成功] 打包完成！
echo ========================================
echo.
echo 可执行文件位置: dist\DAWN_TrainGUI\DAWN_TrainGUI.exe
echo 完整目录: %CD%\dist\DAWN_TrainGUI\
echo.
echo 打包内容说明：
echo - Python 解释器已打包
echo - Train GUI 依赖库已打包
echo - GUI_Utils 与 config_templates 已包含
echo - 运行异常会写入 DAWN_TrainGUI_crash.log（与 exe 同目录）
echo.
pause
