#define AppName "人物视觉实验室"
#define AppVersion "0.2.1"

[Setup]
AppId={{D12E924C-BDA1-41B1-AB3E-2785A39EA918}
AppName={#AppName}
AppVersion={#AppVersion}
DefaultDirName={localappdata}\Programs\PersonVision
DefaultGroupName={#AppName}
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist\installer
OutputBaseFilename=PersonVision-Setup-0.2.1
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\PersonVision.exe

[Files]
Source: "..\dist\PersonVision\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\PersonVision.exe"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\PersonVision.exe"

[Run]
Filename: "{app}\PersonVision.exe"; Description: "启动人物视觉实验室"; Flags: nowait postinstall skipifsilent
