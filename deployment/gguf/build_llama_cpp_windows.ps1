param(
  [Parameter(Mandatory=$true)][string]$LlamaCppDir
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $LlamaCppDir)) {
  throw "llama.cpp dir not found: $LlamaCppDir"
}

Push-Location $LlamaCppDir
try {
  if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    throw "cmake not found in PATH"
  }
  if (-not (Test-Path build)) { New-Item -ItemType Directory -Path build | Out-Null }
  Push-Location build
  try {
    cmake .. 
    cmake --build . --config Release
    Write-Host "Built llama.cpp. Look for build\\bin\\quantize(.exe)."
  } finally {
    Pop-Location
  }
} finally {
  Pop-Location
}

