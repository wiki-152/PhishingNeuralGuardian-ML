# PowerShell setup script for Phish-Defender

# Error handling
$ErrorActionPreference = "Stop"

Write-Host "Phish-Defender Setup" -ForegroundColor Cyan
Write-Host "========================" -ForegroundColor Cyan

# Check Python version
Write-Host "Checking Python version..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version
    Write-Host "Python version: $pythonVersion" -ForegroundColor Green
    
    # Extract version for compatibility checks
    $versionString = $pythonVersion -replace "Python ", ""
    $versionParts = $versionString.Split('.')
    $majorVersion = [int]$versionParts[0]
    $minorVersion = [int]$versionParts[1]
    
    # Store for later use
    $isPython313OrHigher = ($majorVersion -eq 3 -and $minorVersion -ge 13) -or ($majorVersion -gt 3)
} catch {
    Write-Host "Python not found. Please install Python 3.9 or higher." -ForegroundColor Red
    exit 1
}

# Create directory structure
Write-Host "Creating project directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Path "data/raw" -Force | Out-Null
New-Item -ItemType Directory -Path "data/models" -Force | Out-Null
New-Item -ItemType Directory -Path "data/training" -Force | Out-Null
Write-Host "Directories created successfully." -ForegroundColor Green

# Create virtual environment
Write-Host "Creating Python virtual environment..." -ForegroundColor Yellow
python -m venv .venv
Write-Host "Virtual environment created." -ForegroundColor Green

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
try {
    & .\.venv\Scripts\Activate.ps1
    Write-Host "Virtual environment activated." -ForegroundColor Green
} catch {
    Write-Host "Failed to activate virtual environment. You may need to run: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass" -ForegroundColor Red
    exit 1
}

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install required packages
Write-Host "Installing dependencies..." -ForegroundColor Yellow

# Install common packages
python -m pip install numpy pandas scikit-learn vaderSentiment tldextract bs4 lxml deap shap matplotlib seaborn kaggle

# Install TensorFlow with version compatibility
if ($isPython313OrHigher) {
    Write-Host "Detected Python 3.13 or higher - installing latest compatible TensorFlow..." -ForegroundColor Yellow
    python -m pip install tensorflow
} else {
    Write-Host "Installing TensorFlow 2.16.1..." -ForegroundColor Yellow
    python -m pip install tensorflow==2.16.1
}

# Verify installations
Write-Host "Verifying installations..." -ForegroundColor Yellow
python -m pip list

# Check for Kaggle API key
Write-Host "Checking for Kaggle API credentials..." -ForegroundColor Yellow
$kaggleConfigDir = Join-Path $env:USERPROFILE ".kaggle"
$kaggleJsonPath = Join-Path $kaggleConfigDir "kaggle.json"

if (-not (Test-Path $kaggleConfigDir)) {
    New-Item -ItemType Directory -Path $kaggleConfigDir -Force | Out-Null
    Write-Host "Created Kaggle configuration directory: $kaggleConfigDir" -ForegroundColor Green
}

if (-not (Test-Path $kaggleJsonPath)) {
    Write-Host "Kaggle API key not found at $kaggleJsonPath" -ForegroundColor Yellow
    Write-Host "Please set up your Kaggle API credentials to download datasets:" -ForegroundColor Yellow
    Write-Host "1. Go to https://www.kaggle.com/account" -ForegroundColor Yellow
    Write-Host "2. Create a new API token" -ForegroundColor Yellow
    Write-Host "3. Save the downloaded kaggle.json to $kaggleJsonPath" -ForegroundColor Yellow
    
    # Create a placeholder dataset directory
    Write-Host "Creating placeholder dataset directory..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path "data/raw/phishing-email-dataset" -Force | Out-Null
    
    Write-Host "You will need to manually download the dataset from:" -ForegroundColor Yellow
    Write-Host "https://www.kaggle.com/datasets/prashantkumarbaid/phishing-email-dataset" -ForegroundColor Cyan
    Write-Host "And extract it to: ./data/raw/phishing-email-dataset/" -ForegroundColor Cyan
} else {
    Write-Host "Kaggle API credentials found." -ForegroundColor Green
    
    # Download phishing email dataset
    Write-Host "Downloading phishing email dataset from Kaggle..." -ForegroundColor Yellow
    try {
        kaggle datasets download -d "prashantkumarbaid/phishing-email-dataset" -p "data/raw"
        Write-Host "Dataset downloaded successfully." -ForegroundColor Green
        
        # Unzip the dataset
        Write-Host "Extracting dataset..." -ForegroundColor Yellow
        $zipPath = "data/raw/phishing-email-dataset.zip"
        if (Test-Path $zipPath) {
            Expand-Archive -Path $zipPath -DestinationPath "data/raw" -Force
            Remove-Item $zipPath
            Write-Host "Dataset extracted." -ForegroundColor Green
        }
    } catch {
        Write-Host "Failed to download dataset. Please check your Kaggle API key or download manually." -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Red
        Write-Host "You can manually download the dataset from:" -ForegroundColor Yellow
        Write-Host "https://www.kaggle.com/datasets/prashantkumarbaid/phishing-email-dataset" -ForegroundColor Cyan
        Write-Host "And extract it to: ./data/raw/phishing-email-dataset/" -ForegroundColor Cyan
    }
}

# Create requirements.txt for future reference
Write-Host "Generating requirements.txt..." -ForegroundColor Yellow
python -m pip freeze > requirements.txt

Write-Host "Setup complete! Phish-Defender environment is ready." -ForegroundColor Green
Write-Host "To activate this environment in the future, run:" -ForegroundColor Cyan
Write-Host ".\.venv\Scripts\Activate.ps1" -ForegroundColor White 