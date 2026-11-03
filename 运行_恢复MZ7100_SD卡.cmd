@echo off
set "SCRIPT=F:\MLK\DPU_ZYNQ7000\usb_mic_bringup_v1\scripts\host\flash_mz7100_sd_image_admin.ps1"
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "Start-Process powershell.exe -Verb RunAs -ArgumentList '-NoProfile -ExecutionPolicy Bypass -File \"%SCRIPT%\"'"
if errorlevel 1 (
  echo Failed to request administrator privileges.
  pause
)
