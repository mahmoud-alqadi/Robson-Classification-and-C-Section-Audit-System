@echo off
echo ================================================
echo   Robson Automated Reporting System - Demo
echo   WHO Ten-Group Classification
echo ================================================
echo.
echo Sending hospital birth records to the system...
echo.
python send_to_robson.py sample_hospital_data.csv
echo.
pause
