param(
    [ValidateSet('stage1', 'stage2', 'stage3', 'stage4', 'select', 'all')]
    [string]$Stage = 'stage1'
)

# Per-dataset tuning for the datasets that do not yet reach a 0.01 mean MSE
# gain over GTR: ETTm1, ETTm2 (stage1), PEMS03 (stage2), Electricity (stage3),
# Weather (stage4). See docs/APCM_RUN_GUIDE.md, section 7.
# Re-running a stage skips jobs already in results_apcm_mse/runs.jsonl.

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location -LiteralPath $repoRoot

if (-not $env:APCM_OUT) { $env:APCM_OUT = 'results_apcm_mse' }
$python = if ($env:APCM_PYTHON) { $env:APCM_PYTHON } else { 'python' }

# Candidate lists are fixed BEFORE running. The first entry is the score
# reference of the per-dataset selection.
$base = 'GTR-FD5L1,APCM-FD5L1,APCM-CE-FD5L1'
$lossLr = 'GTR-MAE,GTR-HY,GTR-FD8,APCM-CE-FD8,APCM-CE-HY,' +
    'GTR-FD5L1-LR5e4,GTR-FD5L1-LR2e3,APCM-FD5L1-LR5e4,APCM-FD5L1-LR2e3,' +
    'APCM-CE-FD5L1-LR5e4,APCM-CE-FD5L1-LR2e3'
$pemsExtra = 'APCM-FD5L1-K4,APCM-FD5L1-K16,APCM-FD5L1-P8'
$elecExtra = 'GTR-FD5L1-noRevIN,APCM-CE-FD5L1-noRevIN'
# Method extensions A (FC: future-cycle residual) and B (EMA weights).
$methodAB = 'GTR-FD5L1-FC,GTR-FD5L1-EMA,GTR-FD5L1-FC-EMA,' +
    'APCM-CE-FD5L1-FC,APCM-CE-FD5L1-EMA,APCM-CE-FD5L1-FC-EMA'
$methodABPems = 'APCM-FD5L1-FC,APCM-FD5L1-EMA,APCM-FD5L1-FC-EMA,' +
    'APCM-CE-FD5L1-FC,APCM-CE-FD5L1-EMA,APCM-CE-FD5L1-FC-EMA'

function Invoke-ApcmRun {
    param([string[]]$Arguments)
    & $python 'utils/run_apcm_experiments.py' '--stage' 'full' @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Training runner stopped with exit code $LASTEXITCODE" }
}

function Invoke-Select {
    param([string]$Only, [string]$Candidates, [string]$Tag)
    $out = Join-Path $env:APCM_OUT "per_dataset_$Tag.txt"
    & $python 'utils/select_apcm_plus.py' '--per_dataset' '--only' $Only `
        '--candidates' $Candidates '--selection_seeds' '2024,2025,2026' '--tag' $Tag |
        Set-Content -LiteralPath $out -Encoding utf8
    if ($LASTEXITCODE -ne 0) { Write-Warning "Selection for $Tag not complete yet (see SKIP lines)" }
    else { Get-Content -LiteralPath $out }
}

function Invoke-Stage1 {
    # ETTm1, ETTm2: objective, learning-rate, FC and EMA candidates.
    Invoke-ApcmRun @('--only', 'ETTm2,ETTm1', '--variants', "GTR,$base,$lossLr,$methodAB")
    Invoke-Select 'ETTm1,ETTm2' "$base,$lossLr,$methodAB" 'ett'
}

function Invoke-Stage2 {
    # PEMS03: missing baselines + APCM capacity (K=4/16 groups, patch 8).
    Invoke-ApcmRun @('--only', 'PEMS03', '--variants', "GTR,$base,$pemsExtra,$methodABPems")
    Invoke-Select 'PEMS03' "$base,$pemsExtra,$methodABPems" 'pems03'
}

function Invoke-Stage3 {
    # Electricity: missing baselines + RevIN off (GTR-noRevIN is a reference only).
    Invoke-ApcmRun @('--only', 'Electricity', '--variants', "GTR,$base,$elecExtra,$methodAB,GTR-noRevIN")
    Invoke-Select 'Electricity' "$base,$elecExtra,$methodAB" 'electricity'
}

function Invoke-Stage4 {
    # Weather: same objective and learning-rate candidates as stage1.
    Invoke-ApcmRun @('--only', 'Weather', '--variants', "GTR,$base,$lossLr,$methodAB")
    Invoke-Select 'Weather' "$base,$lossLr,$methodAB" 'weather'
}

function Invoke-AllSelections {
    Invoke-Select 'ETTm1,ETTm2' "$base,$lossLr,$methodAB" 'ett'
    Invoke-Select 'PEMS03' "$base,$pemsExtra,$methodABPems" 'pems03'
    Invoke-Select 'Electricity' "$base,$elecExtra,$methodAB" 'electricity'
    Invoke-Select 'Weather' "$base,$lossLr,$methodAB" 'weather'
}

switch ($Stage) {
    'stage1' { Invoke-Stage1 }
    'stage2' { Invoke-Stage2 }
    'stage3' { Invoke-Stage3 }
    'stage4' { Invoke-Stage4 }
    'select' { Invoke-AllSelections }
    'all' { Invoke-Stage1; Invoke-Stage2; Invoke-Stage3; Invoke-Stage4 }
}

Write-Host "Done. Results are in $env:APCM_OUT\ (runs.jsonl, per_dataset_*.txt/csv)"
