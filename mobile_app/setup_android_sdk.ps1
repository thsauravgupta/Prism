$ErrorActionPreference = "Stop"
$ProgressPreference = 'SilentlyContinue'

$SdkDir = "$env:LOCALAPPDATA\Android\Sdk"
$CmdLineToolsPath = "$SdkDir\cmdline-tools\latest"
$ZipUrl = "https://dl.google.com/android/repository/commandlinetools-win-11076708_latest.zip"
$ZipDest = "$env:TEMP\cmdline-tools.zip"

Write-Host "Setting up Android SDK in $SdkDir"

if (-Not (Test-Path -Path $CmdLineToolsPath)) {
    Write-Host "Downloading Command Line Tools..."
    curl.exe -L -o $ZipDest $ZipUrl
    
    Write-Host "Extracting Command Line Tools..."
    Expand-Archive -Path $ZipDest -DestinationPath "$SdkDir\cmdline-tools" -Force
    
    # Rename extracted 'cmdline-tools' to 'latest' so sdkmanager works correctly
    Rename-Item -Path "$SdkDir\cmdline-tools\cmdline-tools" -NewName "latest"
    
    Remove-Item -Path $ZipDest -Force
} else {
    Write-Host "Command Line Tools already installed."
}

# Set Environment Variables for the current session
$env:ANDROID_HOME = $SdkDir
$env:ANDROID_SDK_ROOT = $SdkDir
$env:PATH += ";$CmdLineToolsPath\bin;$SdkDir\platform-tools"

# Accept all licenses
Write-Host "Accepting Android SDK licenses..."
$answers = "y`n" * 50
$answers | & "$CmdLineToolsPath\bin\sdkmanager.bat" --licenses

Write-Host "Installing required SDK packages (platform-tools, platforms;android-35, build-tools)..."
& "$CmdLineToolsPath\bin\sdkmanager.bat" "platform-tools" "platforms;android-35" "build-tools;35.0.0"

Write-Host "`nAndroid SDK setup complete!"
Write-Host "To build the app, run the following in your terminal:"
Write-Host "  `$env:ANDROID_HOME = `"$SdkDir`""
Write-Host "  `$env:ANDROID_SDK_ROOT = `"$SdkDir`""
Write-Host "  ./gradlew assembleDebug"
