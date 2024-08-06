#!/bin/bash

# Check for Python 3 installation
if ! command -v python3 &> /dev/null
then
    echo "Python 3 could not be found. Please install Python 3.7+ before proceeding."
    exit
fi

# Check for pip installation
if ! command -v pip3 &> /dev/null
then
    echo "pip3 could not be found. Please install pip3 before proceeding."
    exit
fi

# Install virtualenv if not installed
if ! command -v virtualenv &> /dev/null
then
    echo "virtualenv not found. Installing virtualenv..."
    pip3 install virtualenv
fi

# Create and activate virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install required Python packages
echo "Installing required packages..."
pip install -r requirements.txt

# Install spaCy language model
echo "Installing spaCy language model..."
python -m spacy download en_core_web_sm

# Check for .env file and prompt for OpenAI API Key if not found
if [ ! -f .env ]; then
    echo "Creating .env file..."
    touch .env
    echo "Please enter your OpenAI API Key:"
    read -s OPENAI_API_KEY
    echo "OPENAI_API_KEY=$OPENAI_API_KEY" >> .env
fi

echo "Installation complete. To start Nyra, run 'source venv/bin/activate' and then 'python main.py'."
