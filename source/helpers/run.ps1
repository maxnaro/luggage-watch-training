<#
.SYNOPSIS
    Automates the build, train, and export workflow using a config.json file.
.EXAMPLE
    .\run.ps1 -Train
.EXAMPLE
    .\run.ps1 -Train -ConfigFile "experiments/config_v2.json"
#>

param(
    [switch]$Build,
    [switch]$Train,
    [switch]$Export,
    [string]$ConfigFile = "config.json",
    [switch]$Help
)

$ErrorActionPreference = "Stop"

# Help Message
if ($Help -or (-not $Build -and -not $Train -and -not $Export)) {
    Write-Host "Luggage Watch Workflow (Config-Based)" -ForegroundColor Yellow
    Write-Host "Usage: .\run.ps1 [OPTIONS]"
    Write-Host ""
    Write-Host "  -Build        Build the Docker image."
    Write-Host "  -Train        Run training using settings from config file."
    Write-Host "  -Export       Export model using settings from config file."
    Write-Host "  -ConfigFile   Path to JSON config (Default: config.json)."
    exit
}

# Load config.json
if (-not (Test-Path $ConfigFile)) {
    Write-Error "Configuration file not found at: $ConfigFile"
}
$Config = Get-Content -Raw $ConfigFile | ConvertFrom-Json
$ProjectName = $Config.projectName

# Build Docker Image
if ($Build) {
    Write-Host "`n[1/3] Building Docker Image ($ProjectName)..." -ForegroundColor Cyan
    docker build -t $ProjectName .
    if ($LASTEXITCODE -ne 0) { Write-Error "Build failed." }
}

# Training Step
if ($Train) {
    Write-Host "`n[2/3] Starting Training..." -ForegroundColor Cyan

    # Ensure runs directory exists
    if (-not (Test-Path $Config.paths.runs)) { New-Item -ItemType Directory -Path $Config.paths.runs | Out-Null }

    # Construct Training Arguments dynamically from JSON
    $TrainArgs = @()
    # Always include data config location (inside container)
    $TrainArgs += "--data", "config/data.yaml"
    
    # Add other keys from the 'train' section of config
    foreach ($key in $Config.train.PSObject.Properties.Name) {
        $value = $Config.train.$key
        $TrainArgs += "--$key", "$value"
    }

    # Convert array to string for display purposes
    Write-Host "Arguments: $TrainArgs" -ForegroundColor DarkGray

    # Run Docker
    docker run --gpus all --ipc=host -it `
        -v "$($Config.paths.data):/app/data" `
        -v "$($PWD)\$($Config.paths.runs):/app/runs" `
        $ProjectName `
        $TrainArgs
}

# Export
if ($Export) {
    Write-Host "`n[3/3] Exporting Model..." -ForegroundColor Cyan

    $RunName = $Config.train.name
    if (-not $RunName) { $RunName = "exp" }

    $ContainerWeightsPath = "/app/runs/$RunName/weights/best.pt"

    $ExportArgs = @()
    $ExportArgs += "--weights", $ContainerWeightsPath
    
    foreach ($key in $Config.export.PSObject.Properties.Name) {
        $value = $Config.export.$key
        # Handle boolean flags (like --simplify) which have no value
        if ($value -is [bool]) {
            if ($value) { $ExportArgs += "--$key" }
        } else {
            $ExportArgs += "--$key", "$value"
        }
    }

    docker run --rm --gpus all `
        -v "$($PWD)\$($Config.paths.runs):/app/runs" `
        -v "$($PWD)\$($Config.paths.model):/app/model" `
        --entrypoint python3 `
        $ProjectName `
        /app/source/export.py $ExportArgs
}

Write-Host "`nDone!" -ForegroundColor Green