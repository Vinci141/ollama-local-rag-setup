@echo off
REM setup_llama_windows.bat
REM Non-technical friendly installer to create a venv, install requirements, install Ollama (manual step), pull a Llama model and run a quick test.

echo ==================================================
echo Llama Project Setup (Windows) - Automated Batch Script
echo ==================================================

:: Check for Python
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python not found in PATH. Please install Python from https://www.python.org/downloads/windows/ and check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

:: Create project folder on Desktop if not present and enter it
set PROJECT_DIR=%USERPROFILE%\Desktop\llama-project
if not exist "%PROJECT_DIR%" (
    mkdir "%PROJECT_DIR%"
)
cd /d "%PROJECT_DIR%"

echo Project folder: %PROJECT_DIR%

:: Create virtual environment
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to create virtual environment. Ensure you have permissions and a working Python installation.
        pause
        exit /b 1
    )
) else (
    echo Virtual environment already exists. Skipping venv creation.
)

:: Activate venv for remainder of the script
call venv\Scripts\activate.bat
if "%VIRTUAL_ENV%"=="" (
    REM On some systems the VIRTUAL_ENV variable may not be set; still continue but warn
    echo Warning: virtual environment activation may have failed. If subsequent steps fail, open a new CMD and run: venv\Scripts\activate
)

:: Create requirements.txt
echo Creating requirements.txt...
>requirements.txt echo faiss-cpu
>>requirements.txt echo numpy
>>requirements.txt echo psutil
>>requirements.txt echo tiktoken
>>requirements.txt echo torch
>>requirements.txt echo bm25s
>>requirements.txt echo PyStemmer
>>requirements.txt echo sentence-transformers
>>requirements.txt echo transformers
>>requirements.txt echo ollama
>>requirements.txt echo PyMuPDF

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
if %ERRORLEVEL% NEQ 0 (
    echo Failed to upgrade pip. Continuing with existing pip.
)

:: Install requirements (first attempt)
echo Installing packages from requirements.txt (this may take several minutes)...
python -m pip install -r requirements.txt
set INSTALL_RC=%ERRORLEVEL%

if %INSTALL_RC% NEQ 0 (
    echo.
    echo Installation encountered errors. Attempting fallback for torch (CPU wheels) and retrying.
    echo Installing CPU-only PyTorch wheels...
    python -m pip install --upgrade pip
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    if %ERRORLEVEL% NEQ 0 (
        echo Fallback PyTorch install failed. Please see https://pytorch.org/get-started/locally/ for manual instructions.
    ) else (
        echo Fallback PyTorch installed successfully. Re-trying requirements installation for remaining packages...
        REM Reinstall other requirements ignoring torch (already installed)
        python -m pip install faiss-cpu numpy psutil tiktoken bm25s PyStemmer sentence-transformers transformers ollama PyMuPDF
    )
)

echo.

:: Check if ollama is installed (CLI)
where ollama >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Ollama CLI not found. Please download and install Ollama for Windows from: https://ollama.com/download
    echo After installing Ollama, re-open CMD and run this script again or run: ollama pull llama2
    pause
    echo Skipping model pull because Ollama is not installed.
) else (
    echo Ollama found. Pulling 'llama2' model (this will download model files and may take time)...
    ollama pull llama2
    if %ERRORLEVEL% NEQ 0 (
        echo Model pull failed or "llama2" not available. Run "ollama list" to see available model names and replace "llama2" accordingly.
    ) else (
        echo Model pulled successfully.
    )
)

:: Create a quick Python test file to verify the setup
>test_ollama.py echo from ollama import pull, chat
>>test_ollama.py echo
>>test_ollama.py echo MODEL = "llama2"
>>test_ollama.py echo 
>>test_ollama.py echo try:
>>test_ollama.py echo ^    pull(MODEL)
>>test_ollama.py echo ^    res = chat(MODEL, "Hello from automated setup. Please reply with a short confirmation message.")
>>test_ollama.py echo ^    print("Model response:\n", res)
>>test_ollama.py echo except Exception as e:
>>test_ollama.py echo ^    print("Test failed:", e)

echo.
echo Setup finished.
if exist test_ollama.py (
    echo You can run a quick test now with: python test_ollama.py
)
echo If Ollama was not installed, please install it from https://ollama.com/download and then run: ollama pull <model-name>
echo Press any key to exit...
pause >nul
