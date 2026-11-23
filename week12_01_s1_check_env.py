"""
第 12 週 · 第一節：量化環境檢查（十項）

用法：
    conda activate nlp_D1234567
    python week12_01_s1_check_env.py

前八項是 GPU 環境的基本檢查（驅動、環境、PyTorch、GPU、sm_120、矩陣相乘、transformers），
再加兩項量化需要的：
    ⑨ bitsandbytes 能在 RTX 5080 上做 4 位元矩陣乘法
    ⑩ 硬碟空間夠（本週模型約 15 GB，另外轉 GGUF 需要約 15 GB 暫存）
沒通過的項目，會印出該打的補救指令（指令也整理在 week12_終端機指令.md）。
"""
import os
import shutil
import subprocess
import sys
import time

results = []


def check(no, name, ok, detail="", fix=""):
    results.append(ok)
    mark = "✅" if ok else "❌"
    print(f"{mark} {no} {name}" + (f"：{detail}" if detail else ""))
    if not ok and fix:
        print(f"      → 補救：{fix}")


print("=" * 64)
print("第 12 週 · 量化環境檢查")
print("=" * 64)

# ① 驅動
drv = None
if shutil.which("nvidia-smi"):
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True)
    if out.returncode == 0 and out.stdout.strip():
        gpu_name, drv, vram = [x.strip() for x in out.stdout.strip().splitlines()[0].split(",")]
check("①", "驅動", drv is not None and int(drv.split(".")[0]) >= 580,
      f"{gpu_name}，驅動 {drv}，顯存 {vram}" if drv else "找不到 nvidia-smi",
      "教室電腦請舉手；自己的電腦請先裝 580 版以上的 NVIDIA 驅動")

# ② 不在 base 環境
env = os.environ.get("CONDA_DEFAULT_ENV", "")
check("②", "conda 環境", bool(env) and env != "base", env or "（沒有在 conda 環境裡）",
      "conda activate nlp_學號（還沒建立：conda create -n nlp_學號 python=3.11 -y）")

# ③～⑦ PyTorch
try:
    import torch
    has_torch = True
except ImportError:
    has_torch = False

if not has_torch:
    for no, name in [("③", "PyTorch"), ("④", "看得到 GPU"), ("⑤", "compute capability"),
                     ("⑥", "認得 sm_120"), ("⑦", "bf16 矩陣相乘")]:
        check(no, name, False, "沒有安裝 PyTorch",
              "pip install torch --index-url https://download.pytorch.org/whl/cu132")
else:
    check("③", "PyTorch", "+cu" in torch.__version__ or (torch.version.cuda is not None),
          f"{torch.__version__}（CUDA {torch.version.cuda}）",
          "pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu132")
    cuda_ok = torch.cuda.is_available()
    check("④", "看得到 GPU", cuda_ok, str(cuda_ok), "確認驅動；PyTorch 要裝 CUDA 版（見③）")
    cap = torch.cuda.get_device_capability(0) if cuda_ok else None
    check("⑤", "compute capability", cap is not None and cap[0] >= 8,
          f"{cap}" + ("（RTX 50 系列 = (12, 0)）" if cap else ""), "這張卡太舊或沒偵測到")
    archs = torch.cuda.get_arch_list() if cuda_ok else []
    need = f"sm_{cap[0]}{cap[1]}" if cap else "sm_120"
    check("⑥", f"PyTorch 認得 {need}", need in archs, " ".join(archs[-4:]),
          "PyTorch 太舊，重裝 cu132 版（見③）")
    try:
        a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(10):
            b = a @ a
        torch.cuda.synchronize()
        ms = (time.time() - t) / 10 * 1000
        check("⑦", "bf16 矩陣相乘", True, f"4096×4096 每次 {ms:.2f} ms（約 {2 * 4096 ** 3 / ms / 1e9:.0f} TFLOPS）")
    except Exception as e:
        check("⑦", "bf16 矩陣相乘", False, f"{type(e).__name__}: {e}", "見③④⑥")

# ⑧ transformers / accelerate
try:
    import accelerate
    import transformers
    ok = int(transformers.__version__.split(".")[0]) >= 5 or transformers.__version__ >= "4.46"
    check("⑧", "transformers、accelerate", ok,
          f"transformers {transformers.__version__}，accelerate {accelerate.__version__}",
          "pip install -U transformers accelerate")
except ImportError as e:
    check("⑧", "transformers、accelerate", False, str(e), "pip install transformers accelerate")

# ⑨ bitsandbytes：真的做一次 4 位元矩陣乘法
try:
    import bitsandbytes as bnb
    import torch
    layer = bnb.nn.Linear4bit(1024, 1024, bias=False, compute_dtype=torch.bfloat16, quant_type="nf4").to("cuda")
    x = torch.randn(4, 1024, device="cuda", dtype=torch.bfloat16)
    y = layer(x)
    torch.cuda.synchronize()
    packed = layer.weight.data
    check("⑨", "bitsandbytes 4 位元", y.shape == (4, 1024),
          f"bitsandbytes {bnb.__version__}；1024×1024 權重存成 {packed.dtype} {tuple(packed.shape)}，"
          f"{packed.numel()} bytes（bf16 要 {1024 * 1024 * 2} bytes）")
except ImportError:
    check("⑨", "bitsandbytes 4 位元", False, "沒有安裝", "pip install -U bitsandbytes")
except Exception as e:
    check("⑨", "bitsandbytes 4 位元", False, f"{type(e).__name__}: {str(e)[:120]}",
          "pip install -U bitsandbytes（要 0.45 以上才支援 RTX 50 系列）")

# ⑩ 硬碟空間
hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
target = hf_home if os.path.exists(hf_home) else os.path.expanduser("~")
free_gb = shutil.disk_usage(target).free / 1e9
check("⑩", "硬碟空間", free_gb >= 20, f"{target} 還剩 {free_gb:.0f} GB（建議 20 GB 以上）",
      "刪掉不用的模型：hf cache ls 看大小，hf cache rm 刪除；或 export HF_HOME=/大硬碟/hf")

# ---------------------------------------------------------------- 總結
n_ok = sum(results)
print("-" * 64)
print(f"結果：{n_ok} / {len(results)} 項通過")
if n_ok == len(results):
    print("環境 OK，可以開始量化了！")
print(f"Python {sys.version.split()[0]}", end="")
if has_torch:
    print(f"｜PyTorch {torch.__version__}｜CUDA {torch.version.cuda}", end="")
try:
    import bitsandbytes
    print(f"｜bitsandbytes {bitsandbytes.__version__}", end="")
except ImportError:
    pass
print()
