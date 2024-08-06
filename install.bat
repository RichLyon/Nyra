@echo off
REM Check for Python 3 installation
python --version 2>nul || (
    echo Python 3 is not installed. Please install Python 3.7+ before proceeding.
    pause
    exit /b
)

REM Check for pip installation
pip --version 2>nul || (
    echo pip is not installed. Please install pip before proceeding.
    pause
    exit /b
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
call venv\Scripts\activate

REM Install required Python packages
echo Installing required packages...
pip install -r requirements.txt

REM Install spaCy language model
echo Installing spaCy language model...
python -m spacy download en_core_web_sm

REM Check for .env file and prompt for OpenAI API Key if not found
if not exist .env (
    echo Creating .env file...
    echo Please enter your OpenAI API Key:
    set /p OPENAI_API_KEY=
    echo OPENAI_API_KEY=%OPENAI_API_KEY% > .env
)

echo Installation complete. To start Nyra, run "venv\Scripts\activate" and then "python main.py".
pause
