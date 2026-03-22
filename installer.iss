; Inno Setup script for HROVER
; Requires Inno Setup 6: https://jrsoftware.org/isinfo.php

#define MyAppName      "HROVER"
#define MyAppVersion   "0.1.0"
#define MyAppPublisher "HROVER"
#define MyAppExeName   "HROVER.exe"
#define MyAppURL       "https://github.com/"

[Setup]
AppId={{A3F2C1B0-4E7D-4A9F-8B2E-1D6C5F0E3A7B}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
; Output
OutputDir=dist
OutputBaseFilename=HROVER_Setup_v{#MyAppVersion}
; Compression
Compression=lzma2/ultra64
SolidCompression=yes
; Appearance
WizardStyle=modern
WizardResizable=yes
; Require 64-bit Windows
ArchitecturesInstallIn64BitMode=x64compatible
; Uninstall
UninstallDisplayName={#MyAppName}
UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; \
  Description: "{cm:CreateDesktopIcon}"; \
  GroupDescription: "{cm:AdditionalIcons}"; \
  Flags: unchecked

[Files]
; Main application bundle produced by PyInstaller
Source: "dist\HROVER\*"; \
  DestDir: "{app}"; \
  Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
; Start Menu
Name: "{group}\{#MyAppName}"; \
  Filename: "{app}\{#MyAppExeName}"; \
  Comment: "Overlay Garmin GPX heart rate data onto video files"

; Uninstall entry in Start Menu
Name: "{group}\Uninstall {#MyAppName}"; \
  Filename: "{uninstallexe}"

; Desktop shortcut (optional, unchecked by default)
Name: "{commondesktop}\{#MyAppName}"; \
  Filename: "{app}\{#MyAppExeName}"; \
  Tasks: desktopicon

[Run]
; Offer to launch after install
Filename: "{app}\{#MyAppExeName}"; \
  Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; \
  Flags: nowait postinstall skipifsilent
