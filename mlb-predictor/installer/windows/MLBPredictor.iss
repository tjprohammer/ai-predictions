#define AppName GetStringFileInfo(AddBackslash(SourcePath) + "dist\\MLBPredictor\\MLBPredictor.exe", "ProductName")
#ifndef AppName
  #define AppName "MLB Predictor"
#endif
#ifndef AppVersion
  #define AppVersion "0.1.0"
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
AppVersion={#AppVersion}
DefaultDirName={localappdata}\Programs\MLBPredictor
DefaultGroupName=MLB Predictor
OutputDir={#OutputDir}
OutputBaseFilename=MLBPredictorSetup
Compression=lzma
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
DisableProgramGroupPage=yes

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\MLB Predictor"; Filename: "{app}\MLBPredictor.exe"
Name: "{autodesktop}\MLB Predictor"; Filename: "{app}\MLBPredictor.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Run]
Filename: "{app}\MLBPredictor.exe"; Description: "Launch MLB Predictor"; Flags: nowait postinstall skipifsilent