@echo off
rem Doppelklick = sofort. Mit Verzoegerung z.B.:  "Bildschirm brechen.bat" -Verzoegerung 10
rem Raus: 6 und 7 gleichzeitig druecken.
start "" powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%~dp0bruch.ps1" %*
