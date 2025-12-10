param(
    [string]$Path
)

if (-not $Path) {
    Write-Error "Please provide the path to the data directory using -Path."
}

# Create persistent volume for data in Docker 
docker volume create luggage-watch-data

Write-Host "Copying data to Docker volume... This may take a moment."
docker run --rm `
  -v "${Path}:/source" `
  -v luggage-watch-data:/dest `
  alpine sh -c "cp -r /source/* /dest/"

Write-Host "Data copy complete."