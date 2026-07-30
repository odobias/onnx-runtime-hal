param (
    [Parameter(Mandatory=$true)][string]$version
)
$name = "whisper-accuracy-samples"

. ..\common\nuget\create_nuget.ps1

$nuspecContent = Get-Content -Path "models.nuspec.template"
$nuspecContent = $nuspecContent -replace "@NAME@", $name
Set-Content -Path "models.nuspec" -Value $nuspecContent

Create-Nuget models.nuspec,models -Platform any -Version $version
Publish-Nuget "$name.$version.nupkg" -Directory $name
