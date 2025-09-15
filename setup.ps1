# PowerShell script for Tiny Children Stories Model Training Setup

# Default configuration
$PROJECT_ROOT = if ($env:PROJECT_ROOT) { $env:PROJECT_ROOT } else { Get-Location }
$VENV_PATH = if ($env:VENV_PATH) { $env:VENV_PATH } else { Join-Path $PROJECT_ROOT "venv" }
$CHECKPOINT_DIR = if ($env:CHECKPOINT_DIR) { $env:CHECKPOINT_DIR } else { Join-Path $PROJECT_ROOT "checkpoints" }
$LORA_CHECKPOINT_DIR = if ($env:LORA_CHECKPOINT_DIR) { $env:LORA_CHECKPOINT_DIR } else { Join-Path $PROJECT_ROOT "lora_checkpoints" }
$REQUIRED_SPACE_MB = if ($env:REQUIRED_SPACE_MB) { [int]$env:REQUIRED_SPACE_MB } else { 1000 }

# Function to print status messages with colors
function Write-Status {
    param([string]$Message)
    Write-Host "[+] $Message" -ForegroundColor Green
}

function Write-Error {
    param([string]$Message)
    Write-Host "[-] $Message" -ForegroundColor Red
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[!] $Message" -ForegroundColor Yellow
}

# Function to handle errors
function Handle-Error {
    param([string]$Message)
    Write-Error $Message
    exit 1
}

# Function to check if a command exists
function Test-CommandExists {
    param([string]$Command)
    try {
        Get-Command $Command -ErrorAction Stop | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

# Function to check disk space
function Test-DiskSpace {
    $drive = (Get-Location).Drive
    $availableSpace = (Get-WmiObject -Class Win32_LogicalDisk -Filter "DeviceID='$($drive.Name)'").FreeSpace
    $availableSpaceMB = [math]::Round($availableSpace / 1MB)
    
    if ($availableSpaceMB -lt $REQUIRED_SPACE_MB) {
        Write-Warning "Low disk space. Only ${availableSpaceMB}MB available, ${REQUIRED_SPACE_MB}MB required."
        return $false
    }
    return $true
}

# Function to check GPU memory
function Test-GpuMemory {
    if (Test-CommandExists "nvidia-smi") {
        try {
            $totalMemory = & nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits
            $freeMemory = & nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits
            $usedMemory = [int]$totalMemory - [int]$freeMemory
            Write-Status "GPU Memory: ${usedMemory}MB used, ${freeMemory}MB free of ${totalMemory}MB total"
        }
        catch {
            Write-Warning "Failed to query GPU memory"
        }
    }
}

# Function to create project structure
function New-ProjectStructure {
    Write-Status "Creating project structure..."
    
    $directories = @(
        (Join-Path $PROJECT_ROOT "src\data"),
        (Join-Path $PROJECT_ROOT "src\model"),
        (Join-Path $PROJECT_ROOT "src\training"),
        $CHECKPOINT_DIR,
        $LORA_CHECKPOINT_DIR
    )
    
    foreach ($dir in $directories) {
        if (!(Test-Path $dir)) {
            try {
                New-Item -ItemType Directory -Path $dir -Force | Out-Null
            }
            catch {
                Handle-Error "Failed to create directory: $dir"
            }
        }
    }
}

# Function to setup virtual environment
function Initialize-VirtualEnv {
    Write-Status "Creating virtual environment..."
    
    try {
        & python -m venv $VENV_PATH
        if ($LASTEXITCODE -ne 0) {
            Handle-Error "Failed to create virtual environment"
        }
    }
    catch {
        Handle-Error "Failed to create virtual environment"
    }
    
    # Activate virtual environment
    $activateScript = Join-Path $VENV_PATH "Scripts\Activate.ps1"
    if (Test-Path $activateScript) {
        & $activateScript
    }
    else {
        Handle-Error "Failed to find virtual environment activation script"
    }
    
    Write-Status "Installing dependencies..."
    try {
        & python -m pip install --upgrade pip
        & pip install -r requirements.txt
        if ($LASTEXITCODE -ne 0) {
            Handle-Error "Failed to install requirements"
        }
    }
    catch {
        Handle-Error "Failed to install requirements"
    }
}

# Function to prepare dataset
function Initialize-Dataset {
    Write-Status "Preparing dataset..."
    Set-Location $PROJECT_ROOT
    
    # Create a Python script to process the data
    $pythonScript = @'
import os
import sys

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.data_processor import DataProcessor

def main():
    print("[+] Processing dataset into binary files...")
    processor = DataProcessor()
    processor.prepare_dataset()
    print("[+] Data processing completed successfully!")

if __name__ == "__main__":
    main()
'@
    
    $pythonScript | Out-File -FilePath "process_data.py" -Encoding utf8
    
    # Run the data processing script
    try {
        & python process_data.py
        if ($LASTEXITCODE -ne 0) {
            Handle-Error "Failed to process dataset"
        }
    }
    catch {
        Handle-Error "Failed to process dataset"
    }
    
    # Verify the files were created
    $trainFile = Join-Path $PROJECT_ROOT "src\data\train.bin"
    $validationFile = Join-Path $PROJECT_ROOT "src\data\validation.bin"
    
    if (!(Test-Path $trainFile) -or !(Test-Path $validationFile)) {
        Handle-Error "Data processing failed - required files not created"
    }
}

# Function to train base model
function Start-BaseModelTraining {
    Write-Status "Starting base model training..."
    Set-Location $PROJECT_ROOT
    
    $batchSize = if ($env:BATCH_SIZE) { $env:BATCH_SIZE } else { "12" }
    $maxIters = if ($env:MAX_ITERS) { $env:MAX_ITERS } else { "20000" }
    $evalInterval = if ($env:EVAL_INTERVAL) { $env:EVAL_INTERVAL } else { "1000" }
    $evalIters = if ($env:EVAL_ITERS) { $env:EVAL_ITERS } else { "200" }
    $learningRate = if ($env:LEARNING_RATE) { $env:LEARNING_RATE } else { "6e-4" }
    $weightDecay = if ($env:WEIGHT_DECAY) { $env:WEIGHT_DECAY } else { "0.1" }
    $warmupIters = if ($env:WARMUP_ITERS) { $env:WARMUP_ITERS } else { "2000" }
    $lrDecayIters = if ($env:LR_DECAY_ITERS) { $env:LR_DECAY_ITERS } else { "20000" }
    $minLr = if ($env:MIN_LR) { $env:MIN_LR } else { "6e-5" }
    
    try {
        & python src\run_training.py `
            --batch-size $batchSize `
            --max-iters $maxIters `
            --eval-interval $evalInterval `
            --eval-iters $evalIters `
            --learning-rate $learningRate `
            --weight-decay $weightDecay `
            --warmup-iters $warmupIters `
            --lr-decay-iters $lrDecayIters `
            --min-lr $minLr `
            --checkpoint-dir $CHECKPOINT_DIR
            
        if ($LASTEXITCODE -ne 0) {
            Handle-Error "Base model training failed"
        }
    }
    catch {
        Handle-Error "Base model training failed"
    }
}

# Function to perform LoRA finetuning
function Start-LoRAFinetuning {
    do {
        $doFinetune = Read-Host "Do you want to perform LoRA finetuning? (y/n)"
        switch ($doFinetune.ToLower()) {
            { $_ -in @('y', 'yes') } {
                Write-Status "Starting LoRA finetuning..."
                Set-Location $PROJECT_ROOT
                
                $baseModel = Join-Path $CHECKPOINT_DIR "Tiny-Children-Stories-Collection-model.pt"
                $loraBatchSize = if ($env:LORA_BATCH_SIZE) { $env:LORA_BATCH_SIZE } else { "8" }
                $loraLearningRate = if ($env:LORA_LEARNING_RATE) { $env:LORA_LEARNING_RATE } else { "1e-4" }
                $loraMaxIters = if ($env:LORA_MAX_ITERS) { $env:LORA_MAX_ITERS } else { "5000" }
                
                try {
                    & python src\finetune_lora.py `
                        --base-model $baseModel `
                        --batch-size $loraBatchSize `
                        --learning-rate $loraLearningRate `
                        --max-iters $loraMaxIters `
                        --checkpoint-dir $LORA_CHECKPOINT_DIR
                        
                    if ($LASTEXITCODE -ne 0) {
                        Handle-Error "LoRA finetuning failed"
                    }
                }
                catch {
                    Handle-Error "LoRA finetuning failed"
                }
                return
            }
            { $_ -in @('n', 'no') } {
                Write-Status "Skipping LoRA finetuning..."
                return
            }
            default {
                Write-Host "Please answer 'y' or 'n'"
            }
        }
    } while ($true)
}

# Function to test the trained model
function Test-TrainedModel {
    do {
        $doTest = Read-Host "Do you want to test the trained model? (y/n)"
        switch ($doTest.ToLower()) {
            { $_ -in @('y', 'yes') } {
                Write-Status "Testing the trained model..."
                Set-Location $PROJECT_ROOT
                
                $modelPath = Join-Path $CHECKPOINT_DIR "Tiny-Children-Stories-Collection-model.pt"
                $prompts = @(
                    "Once upon a time",
                    "In a magical forest",
                    "The little robot",
                    "The brave knight"
                )
                
                foreach ($prompt in $prompts) {
                    Write-Status "Testing with prompt: '$prompt'"
                    try {
                        & python src\generate.py `
                            --model_path $modelPath `
                            --prompt $prompt `
                            --max_tokens 100 `
                            --temperature 0.8 `
                            --top_k 40
                    }
                    catch {
                        Write-Warning "Failed to test with prompt: '$prompt'"
                    }
                    Write-Host ""
                }
                return
            }
            { $_ -in @('n', 'no') } {
                Write-Status "Skipping model testing..."
                return
            }
            default {
                Write-Host "Please answer 'y' or 'n'"
            }
        }
    } while ($true)
}

# Main execution function
function Main {
    # Check Python installation
    if (!(Test-CommandExists "python")) {
        Handle-Error "Python is not installed. Please install Python 3.8 or higher."
    }
    
    # Check disk space
    if (!(Test-DiskSpace)) {
        Handle-Error "Insufficient disk space"
    }
    
    # Create project structure
    New-ProjectStructure
    
    # Check for CUDA availability
    if (Test-CommandExists "nvidia-smi") {
        try {
            $gpuName = & nvidia-smi --query-gpu=name --format=csv,noheader
            Write-Status "CUDA is available. Using GPU: $gpuName"
            $env:DEVICE = "cuda"
            Test-GpuMemory
        }
        catch {
            Write-Warning "CUDA found but failed to query GPU information"
            $env:DEVICE = "cpu"
        }
    }
    else {
        Write-Warning "CUDA not found. Training will be slower on CPU."
        $env:DEVICE = "cpu"
    }
    
    # Setup virtual environment and dependencies
    Initialize-VirtualEnv
    
    # Start the training process
    Write-Status "Starting training process..."
    $startTime = Get-Date
    
    # Prepare and process dataset
    Initialize-Dataset
    
    # Train base model
    Start-BaseModelTraining
    
    # Perform LoRA finetuning if requested
    Start-LoRAFinetuning
    
    # Test the trained model
    Test-TrainedModel
    
    # Calculate and display total time
    $endTime = Get-Date
    $duration = $endTime - $startTime
    $hours = [math]::Floor($duration.TotalHours)
    $minutes = [math]::Floor($duration.Minutes)
    $seconds = [math]::Floor($duration.Seconds)
    
    Write-Status "Training process completed!"
    Write-Status "Total time: ${hours}h ${minutes}m ${seconds}s"
    
    # Deactivate virtual environment
    if (Get-Command "deactivate" -ErrorAction SilentlyContinue) {
        deactivate
    }
    
    Write-Status "Setup and training completed successfully!"
    Write-Status "You can find the trained models in:"
    Write-Host "  - Base model: $(Join-Path $CHECKPOINT_DIR 'Tiny-Children-Stories-Collection-model.pt')"
    Write-Host "  - LoRA model: $(Join-Path $LORA_CHECKPOINT_DIR 'Tiny-Children-Stories-Collection-LoRA-model.pt')"
    
    # Print final GPU memory status
    if (Test-CommandExists "nvidia-smi") {
        Write-Status "Final GPU Memory Status:"
        & nvidia-smi
    }
}

# Run main function
Main
