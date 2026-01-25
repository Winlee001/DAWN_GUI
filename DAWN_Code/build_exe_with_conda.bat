@echo off
chcp 65001 >nul
echo ========================================
echo DAWN GUI 打包脚本（Conda 环境版本）
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
if not exist "%PYTHON_EXE%" (
    if not exist "%CONDA_ENV%\python.exe" (
        echo [错误] 无法找到 Python 解释器
        pause
        exit /b 1
    ) else (
        set PYTHON_EXE=%CONDA_ENV%\python.exe
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

REM 检查是否存在 .spec 文件
if exist DAWN_GUI.spec (
    echo [信息] 使用 DAWN_GUI.spec 配置文件
    echo [信息] 开始打包（这可能需要 10-30 分钟）...
    echo.
    "%PYTHON_EXE%" -m PyInstaller DAWN_GUI.spec
) else (
    echo [警告] 未找到 DAWN_GUI.spec 文件，使用命令行参数打包...
    echo [信息] 开始打包（这可能需要 10-30 分钟）...
    echo.
    "%PYTHON_EXE%" -m PyInstaller ^
        --name=DAWN_GUI ^
        --icon=GUI_Utils/icon.png ^
        --add-data "GUI_Utils;GUI_Utils" ^
        --hidden-import=PyQt6.QtCore ^
        --hidden-import=PyQt6.QtGui ^
        --hidden-import=PyQt6.QtWidgets ^
        --hidden-import=torch ^
        --hidden-import=torchvision ^
        --hidden-import=numpy ^
        --hidden-import=tifffile ^
        --hidden-import=omegaconf ^
        --hidden-import=psutil ^
        --hidden-import=pynvml ^
        --hidden-import=scipy ^
        --hidden-import=cv2 ^
        --hidden-import=PIL ^
        --hidden-import=skimage ^
        --console ^
        DAWN_GUI.py
)

if errorlevel 1 (
    echo.
    echo [错误] 打包失败！请查看上面的错误信息
    pause
    exit /b 1
)

REM 检查打包结果
if not exist "dist\DAWN_GUI\DAWN_GUI.exe" (
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
echo 可执行文件位置: dist\DAWN_GUI\DAWN_GUI.exe
echo 完整目录: %CD%\dist\DAWN_GUI\
echo.
echo 打包内容说明：
echo - Python 解释器已打包
echo - 所有依赖库已打包
echo - GUI_Utils 资源文件已包含
echo.
echo 注意事项：
echo 1. 打包的文件可能很大（500MB-2GB），这是正常的
echo 2. 如果使用 GPU 版本的 PyTorch，用户仍需要安装 CUDA 运行时库
echo 3. 首次运行可能会比较慢（初始化过程）
echo 4. 可以将整个 dist\DAWN_GUI 目录压缩后分发给用户
echo.
pause