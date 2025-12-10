param(
    [string]$Volume
    [string]$Path
)

if (-not $Path) {
    Write-Error "Please provide the path to the data directory using -Path."
}

if (-not (Test-Path $Path)) {
    Write-Error "The specified path '$Path' does not exist."
}

if (-not $Volume) {
    Write-Error "Please provide the name for the Docker volume using -Volume."
}

# Create persistent volume for data in Docker 
docker volume create $Volume

Write-Host "Copying data to Docker volume... This may take a moment."
docker run --rm `
  -v "${Path}:/source" `
  -v "${Volume}:/dest" `
  alpine sh -c "cp -r /source/* /dest/"

Write-Host "Data copy complete."