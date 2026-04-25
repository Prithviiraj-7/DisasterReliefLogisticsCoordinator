param(
    [string]$TaskId = "task_hard_region",
    [int]$Port = 7860,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

function Test-Command($Name) {
    return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

if (-not (Test-Command "py")) {
    Write-Error "Python launcher 'py' was not found. Install Python and retry."
}

if (-not $SkipInstall) {
    Write-Host "Installing/updating dependencies..."
    py -m pip install --upgrade pip
    py -m pip install -r requirements.txt
}

$baseUrl = "http://127.0.0.1:$Port"
$serverArgs = @("-m", "uvicorn", "server.app:app", "--host", "127.0.0.1", "--port", "$Port")

Write-Host "Starting local server on port $Port..."
$serverProcess = Start-Process -FilePath "py" -ArgumentList $serverArgs -PassThru -WindowStyle Hidden

try {
    $healthOk = $false
    for ($i = 0; $i -lt 20; $i++) {
        Start-Sleep -Milliseconds 500
        try {
            $health = Invoke-RestMethod -Method Get -Uri "$baseUrl/health" -TimeoutSec 5
            if ($health.status -eq "ok") {
                $healthOk = $true
                break
            }
        } catch {
            # Server may still be booting.
        }
    }

    if (-not $healthOk) {
        throw "Server health check failed on $baseUrl/health."
    }

    Write-Host "Health check passed."

    $resetBody = @{ task_id = $TaskId } | ConvertTo-Json
    $reset = Invoke-RestMethod -Method Post -Uri "$baseUrl/reset" -ContentType "application/json" -Body $resetBody
    Write-Host ("Reset passed. done={0}, task={1}" -f $reset.done, $reset.observation.task_id)

    $stepBody = @{ action = @{ command = "get_status" } } | ConvertTo-Json -Depth 5
    $step = Invoke-RestMethod -Method Post -Uri "$baseUrl/step" -ContentType "application/json" -Body $stepBody
    Write-Host ("Step passed. tick={0}, reward={1}, score={2}" -f $step.observation.tick, $step.reward, $step.observation.task_score)

    Write-Host "Local smoke test completed successfully."
}
finally {
    if ($null -ne $serverProcess -and -not $serverProcess.HasExited) {
        Write-Host "Stopping local server..."
        Stop-Process -Id $serverProcess.Id -Force
    }
}
