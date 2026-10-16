@echo off
echo ==========================================
echo STARTING SEQUENTIAL TRAINING (4 BIOMES)
echo ==========================================

echo.
echo [1/4] Training AMAZONIA...
python -m src.train --biome amazonia --epochs 50 --batch_size 4

echo.
echo [2/4] Training CERRADO...
python -m src.train --biome cerrado --epochs 50 --batch_size 4

echo.
echo [3/4] Training MATA ATLANTICA...
python -m src.train --biome mata_atlantica --epochs 50 --batch_size 4

echo.
echo [4/4] Training PANTANAL...
python -m src.train --biome pantanal --epochs 50 --batch_size 4

echo.
echo ==========================================
echo ALL SESSIONS FINISHED
echo Check the 'results' and 'checkpoints' folders
echo ==========================================
pause
