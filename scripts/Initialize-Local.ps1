[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$envPath = Join-Path $projectRoot '.env'
if (Test-Path -LiteralPath $envPath) {
    Write-Host 'Existing .env retained. No credentials changed.'
    exit 0
}
function New-LocalSecret {
    $bytes = New-Object byte[] 32
    $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    try { $rng.GetBytes($bytes) } finally { $rng.Dispose() }
    return ([BitConverter]::ToString($bytes)).Replace('-', '').ToLowerInvariant()
}
$databasePassword = New-LocalSecret
$apiKey = New-LocalSecret
$databaseUrl = "postgresql://memory_optimizer:${databasePassword}@127.0.0.1:55433/memory_optimizer"
$lines = @(
    "POSTGRES_PASSWORD=$databasePassword",
    "MEMORY_DATABASE_URL=$databaseUrl",
    "MEMORY_TEST_DATABASE_URL=$databaseUrl",
    "MEMORY_API_KEY=$apiKey",
    'MEMORY_DATABASE_SCHEMA=memory_optimizer'
)
# CreateNew prevents overwriting a file created by another initializer concurrently.
$stream = [System.IO.File]::Open($envPath, [System.IO.FileMode]::CreateNew)
try {
    $content = [System.Text.UTF8Encoding]::new($false).GetBytes(($lines -join "`n") + "`n")
    $stream.Write($content, 0, $content.Length)
} finally { $stream.Dispose() }
Write-Host 'Created private .env with local development credentials. Values were not printed.'
