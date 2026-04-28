<#
Registers the daily mean-reversion live runner with Windows Task Scheduler.

Trigger: daily at 02:00 local Sydney time (= 16:00 UTC AEST winter / 15:00 UTC AEDT summer).
Repeats every 15 minutes for 6 hours.

The 02:00 + 6h window was chosen so the script's internal trade-window check
will fire inside the MOC submission window for every US market session,
including early-close days (13:00 ET), across all four AU/US DST combinations.

Most invocations exit fast at the "too early / too late / already ran today"
check; only the one that lands inside the dynamic ET trade window submits orders.

Re-run this script if the python.exe path changes or you want to update the
schedule. It will replace any existing task with the same name.

Run as Administrator (required so the task can run while the user is logged out).
#>

[CmdletBinding()]
param(
    [string]$TaskName = "MeanReversionPaperTrade",
    [string]$ProjectDir = (Resolve-Path "$PSScriptRoot\..").Path,
    [string]$PythonExe,
    [string]$LocalStartTime = "02:00",
    [int]$RepeatMinutes = 15,
    [int]$DurationHours = 6
)

$ErrorActionPreference = "Stop"

if (-not $PythonExe) {
    $PythonExe = (Get-Command python.exe -ErrorAction SilentlyContinue).Source
    if (-not $PythonExe) {
        throw "Could not find python.exe on PATH. Pass -PythonExe 'C:\path\to\python.exe' explicitly."
    }
}

$LiveScript = Join-Path $ProjectDir "live.py"
if (-not (Test-Path $LiveScript)) {
    throw "live.py not found at $LiveScript"
}

Write-Host "Registering scheduled task '$TaskName'"
Write-Host "  python      : $PythonExe"
Write-Host "  live.py     : $LiveScript"
Write-Host "  starts at   : $LocalStartTime local"
Write-Host "  repeat      : every $RepeatMinutes min for $DurationHours hours"

$action = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument "`"$LiveScript`"" `
    -WorkingDirectory $ProjectDir

$trigger = New-ScheduledTaskTrigger -Daily -At $LocalStartTime
$trigger.Repetition = (New-ScheduledTaskTrigger `
    -Once -At $LocalStartTime `
    -RepetitionInterval (New-TimeSpan -Minutes $RepeatMinutes) `
    -RepetitionDuration (New-TimeSpan -Hours $DurationHours)).Repetition

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -WakeToRun `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 5) `
    -MultipleInstances IgnoreNew

$principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType S4U `
    -RunLevel Highest

if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Write-Host "Removing existing task '$TaskName'"
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "Mean-reversion daily MOC paper-trade runner. See live.py."

Write-Host ""
Write-Host "Done. Verify with:  Get-ScheduledTask -TaskName '$TaskName' | Format-List *"
Write-Host "Run on demand with: Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "Remove with:        Unregister-ScheduledTask -TaskName '$TaskName' -Confirm:`$false"
