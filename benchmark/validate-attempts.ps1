[CmdletBinding()]
param(
    [string]$Ledger = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
if (-not $Ledger) { $Ledger = Join-Path $root "results\ledgers\attempts.jsonl" }
if (-not (Test-Path -LiteralPath $Ledger)) { throw "Attempt ledger does not exist: $Ledger" }

$schemaPath = Join-Path $PSScriptRoot "schemas\attempt-record.schema.json"
$schema = Get-Content -LiteralPath $schemaPath -Raw -Encoding UTF8 | ConvertFrom-Json
$allowed = @($schema.properties.PSObject.Properties.Name)
$required = @($schema.required)
$errors = [Collections.Generic.List[string]]::new()
$count = 0

foreach ($line in Get-Content -LiteralPath $Ledger -Encoding UTF8) {
    if (-not $line.Trim()) { continue }
    ++$count
    try {
        $record = $line | ConvertFrom-Json -ErrorAction Stop
    } catch {
        $errors.Add("line ${count}: invalid JSON: $($_.Exception.Message)")
        continue
    }

    $fields = @($record.PSObject.Properties.Name)
    foreach ($field in $required) {
        if ($fields -notcontains $field) {
            $errors.Add("line ${count}: missing required field '$field'")
        }
    }
    foreach ($field in $fields) {
        if ($allowed -notcontains $field) {
            $errors.Add("line ${count}: unexpected field '$field'")
        }
    }
    foreach ($field in $allowed) {
        $propertySchema = $schema.properties.$field
        if ($fields -notcontains $field) { continue }
        if (($propertySchema.PSObject.Properties.Name -contains "const") -and
            $record.$field -ne $propertySchema.const) {
            $errors.Add("line ${count}: invalid $field '$($record.$field)'")
        }
        if ($propertySchema.enum -and @($propertySchema.enum) -notcontains $record.$field) {
            $errors.Add("line ${count}: invalid $field '$($record.$field)'")
        }
        if ($propertySchema.minLength -and ([string]$record.$field).Length -lt $propertySchema.minLength) {
            $errors.Add("line ${count}: $field is shorter than $($propertySchema.minLength)")
        }
    }

    $timestamp = [DateTimeOffset]::MinValue
    if ($fields -contains "timestamp_utc" -and
        -not [DateTimeOffset]::TryParse([string]$record.timestamp_utc, [ref]$timestamp)) {
        $errors.Add("line ${count}: invalid timestamp_utc '$($record.timestamp_utc)'")
    }
    if ($fields -contains "model_compilation_provenance_ids") {
        $ids = @($record.model_compilation_provenance_ids)
        if (@($ids | Sort-Object -Unique).Count -ne $ids.Count) {
            $errors.Add("line ${count}: duplicate model compilation provenance ID")
        }
    }
}

if (-not $count) { throw "Attempt ledger is empty: $Ledger" }
if ($errors.Count) {
    $errors | ForEach-Object { Write-Error $_ }
    exit 1
}
Write-Host "Attempt ledger valid: $count records" -ForegroundColor Green
