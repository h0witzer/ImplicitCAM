@echo off
rem Opens a STEP part alone in icam-sim's part view, drawn from its exact signed distance.
rem Double-click: THE PART. Drop any .step or .stp file on this file: that part.
rem The first run builds the viewer (a few minutes); the window shows the part coarse
rem within seconds and sharpens as finer levels come in.
setlocal
set "ROOT=%~dp0.."
set "FILE=%~1"
if "%FILE%"=="" set "FILE=%ROOT%\test-parts\the-part\the-part-emm0sh.step"
if not exist "%FILE%" (
  echo No part found. Drop a .step file on this file.
  pause
  exit /b 1
)
pushd "%ROOT%"
cargo build --release -p icam-sim || (popd & pause & exit /b 1)
"target\release\icam-sim.exe" part "%FILE%"
if errorlevel 1 pause
popd
