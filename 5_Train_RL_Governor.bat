@echo off
echo "================================="
echo "  Activating Virtual Environment"
echo "================================="
call venv\Scripts\activate.bat

echo "================================="
echo "  Training RL Governor Model"
echo "================================="
python train_rl_governor.py
pause