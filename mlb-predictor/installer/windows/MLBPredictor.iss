#ifndef AppName
  #define AppName "MLB Predictor"
#endif
#ifndef AppVersion
  #define AppVersion "0.8.0-beta"
#endif
#ifndef AppPublisher
  #define AppPublisher "TJ Prohammer"
#endif
#ifndef AppPublisherURL
  #define AppPublisherURL "https://github.com/tjprohammer/ai-predictions"
#endif
#ifndef AppSupportURL
  #define AppSupportURL "https://github.com/tjprohammer/ai-predictions/issues"
#endif
#ifndef AppUpdatesURL
  #define AppUpdatesURL "https://github.com/tjprohammer/ai-predictions/releases"
#endif
#ifndef AppExeName
  #define AppExeName "MLBPredictor.exe"
#endif
#ifndef SetupIconFile
  #define SetupIconFile AddBackslash(SourcePath) + "mlb_predictor.ico"
#endif
#ifndef AppInstallDirName
  #define AppInstallDirName "MLBPredictor"
#endif
#ifndef OutputBaseFilename
  #define OutputBaseFilename AppInstallDirName + "-Windows-v" + AppVersion + "-Setup"
#endif
#ifndef VersionInfoProductVersion
  #define VersionInfoProductVersion "0.8.0.0"
#endif
#ifndef SourceDir
  #define SourceDir AddBackslash(SourcePath) + "dist\\MLBPredictor"
#endif
#ifndef OutputDir
  #define OutputDir AddBackslash(SourcePath) + "release"
#endif

[Setup]
AppId={{9F8C6E26-2F64-4F01-8E51-83B58D2F85B1}
AppName={#AppName}
AppVerName={#AppName} {#AppVersion}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppPublisherURL}
AppSupportURL={#AppSupportURL}
AppUpdatesURL={#AppUpdatesURL}
DefaultDirName={localappdata}\Programs\{#AppInstallDirName}
DefaultGroupName={#AppName}
OutputDir={#OutputDir}
OutputBaseFilename={#OutputBaseFilename}
Compression=lzma
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\{#AppExeName}
SetupIconFile={#SetupIconFile}
VersionInfoCompany={#AppPublisher}
VersionInfoDescription={#AppName} desktop installer
VersionInfoProductName={#AppName}
VersionInfoProductVersion={#VersionInfoProductVersion}

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#SetupIconFile}"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"; IconFilename: "{app}\mlb_predictor.ico"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; IconFilename: "{app}\mlb_predictor.ico"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent