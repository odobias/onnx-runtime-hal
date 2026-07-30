setlocal
cd /d "%~dp0"

powershell ./download.ps1 || exit /b 1

call ..\common\nuget\publish.cmd %~1 || exit /b 1
endlocal
