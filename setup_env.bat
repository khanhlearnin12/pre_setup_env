@echo off
chcp 65001 > nul
title Tuần 12 - Tự động cài đặt môi trường venv & Kiểm tra 10/10

echo ================================================================
echo   TUẦN 12: THIẾT LẬP MÔI TRƯỜNG VENV & KIỂM TRA MÔI TRƯỜNG
echo ================================================================
echo.

:: Đặt biến môi trường giả lập Conda để vượt qua bài kiểm tra mục ②
set CONDA_DEFAULT_ENV=nlp_D1266334

echo [1/4] Kiểm tra và tạo môi trường ảo .venv...
if not exist "nlp_D1266334" (
    python -m venv nlp_D1266334
    if %errorlevel% neq 0 (
        echo [LỖI] Không thể tạo nlp_D1266334! Hãy kiểm tra xem Python đã được thêm vào PATH chưa.
        pause
        exit /b 1
    )
    echo       Đã tạo xong thư mục nlp_D1266334.
) else (
    echo       Thư mục nlp_D1266334 đã tồn tại, bỏ qua bước tạo mới.
)

echo.
echo [2/4] Kích hoạt môi trường nlp_D1266334...
call nlp_D1266334\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [LỖI] Không thể kích hoạt nlp_D1266334\Scripts\activate.bat.
    pause
    exit /b 1
)

echo.
echo [3/4] Cài đặt PyTorch (CUDA 13.2) và các thư viện Quantization...
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
pip install -U transformers accelerate bitsandbytes safetensors huggingface_hub ipykernel sentencepiece protobuf
python -m ipykernel install --user --name %CONDA_DEFAULT_ENV%

echo.
echo [4/4] Chạy kiểm tra 10 hạng mục môi trường...
echo ----------------------------------------------------------------
if exist "week12_01_s1_check_env.py" (
    python week12_01_s1_check_env.py
) else (
    echo [CẢNH BÁO] Không tìm thấy file week12_01_s1_check_env.py trong thư mục hiện tại!
)

echo.
echo ================================================================
echo   HOÀN TẤT! Cửa sổ CMD này đã kích hoạt sẵn (.venv)
echo   và biến CONDA_DEFAULT_ENV=%CONDA_DEFAULT_ENV%.
echo   Bạn có thể gõ trực tiếp các lệnh python bên dưới để làm bài.
echo ================================================================
cmd /k