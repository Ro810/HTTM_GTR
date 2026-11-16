param(
    [ValidateSet('stage1', 'stage2', 'stage3', 'all')]
    [string]$Stage = 'all'
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location -LiteralPath $repoRoot

if (-not $env:APCM_OUT) {
    $env:APCM_OUT = 'results_apcm_mse'
}
$python = if ($env:APCM_PYTHON) { $env:APCM_PYTHON } else { 'python' }
$newVariants = 'GTR-FD5L1,APCM-FD5L1,APCM-CE-FD5L1'

function Invoke-ApcmRun {
    param([string[]]$Arguments)
    & $python 'utils/run_apcm_experiments.py' '--stage' 'full' @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Training runner stopped with exit code $LASTEXITCODE"
    }
}

function Invoke-Stage1 {
    # One seed, shortest/longest horizon, all non-Traffic long-series data.
    Invoke-ApcmRun @('--only', 'ETTh1,ETTh2,ETTm1,ETTm2,Weather,Solar,Electricity',
        '--pred_lens', '96,720', '--seeds', '2024',
        '--variants', "GTR,APCM,$newVariants")
    Invoke-ApcmRun @('--only', 'PEMS03,PEMS04,PEMS07,PEMS08',
        '--pred_lens', '12,96', '--seeds', '2024', '--variants', $newVariants)
}

function Invoke-Stage2 {
    # Full 4 horizons x 3 seeds, still excluding slow Traffic.
    Invoke-ApcmRun @('--only', 'ETTh1,ETTh2,ETTm1,ETTm2,Weather,Solar',
        '--variants', "GTR,APCM,$newVariants")
    Invoke-ApcmRun @('--only', 'PEMS03,PEMS04,PEMS08,Electricity',
        '--variants', $newVariants)
    Invoke-ApcmRun @('--only', 'PEMS07', '--variants', $newVariants)
}

function Invoke-Stage3 {
    # Slow Traffic is intentionally last.
    Invoke-ApcmRun @('--only', 'Traffic',
        '--variants', 'APCM-FD5L1,APCM-CE-FD5L1')
    Invoke-ApcmRun @('--only', 'Traffic', '--variants', 'GTR-FD5L1')
}

switch ($Stage) {
    'stage1' { Invoke-Stage1 }
    'stage2' { Invoke-Stage2 }
    'stage3' { Invoke-Stage3 }
    'all' { Invoke-Stage1; Invoke-Stage2; Invoke-Stage3 }
}

& $python 'utils/summarize_apcm.py' |
    Set-Content -LiteralPath (Join-Path $env:APCM_OUT 'summary.txt') -Encoding utf8
if ($LASTEXITCODE -ne 0) { throw 'Summary generation failed' }

$selectionArgs = @('utils/select_apcm_plus.py', '--loss', 'FD5L1')
if ($Stage -ne 'stage1') {
    $selectionArgs += @('--selection_seeds', '2024,2025,2026')
}
& $python @selectionArgs |
    Set-Content -LiteralPath (Join-Path $env:APCM_OUT 'apcm_plus.txt') -Encoding utf8
if ($LASTEXITCODE -ne 0) { throw 'APCM+ selection failed' }

Write-Host "Done. Corrected results are in $env:APCM_OUT\"
