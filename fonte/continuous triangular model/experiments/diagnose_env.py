#!/usr/bin/env python3
"""Isolate which native library raises SIGILL on this machine.

A SIGILL kills the whole process, so each check runs in its own subprocess and
the parent survives to report which step failed. Run from the project root:

    python -m experiments.diagnose_env

Exit code is non-zero if any check fails.
"""
from __future__ import annotations

import platform
import subprocess
import sys


CHECKS = {
    "numpy import": "import numpy; print(numpy.__version__)",
    "numpy fft": "import numpy as np; np.fft.rfft(np.random.randn(360)); print('fft ok')",
    "scipy import": "import scipy; print(scipy.__version__)",
    "scipy.signal.stft": (
        "import numpy as np; from scipy.signal import stft; "
        "stft(np.random.randn(360), nperseg=64, noverlap=48); print('stft ok')"
    ),
    "pywt.cwt": (
        "import numpy as np, pywt; "
        "pywt.cwt(np.random.randn(360), np.arange(1, 49), 'morl'); print('cwt ok')"
    ),
    "PIL resize": (
        "import numpy as np; from PIL import Image; "
        "Image.fromarray(np.random.rand(96, 96).astype('float32')).resize((96, 96)); "
        "print('pil ok')"
    ),
    "torch import": "import torch; print(torch.__version__)",
    "torch forward": (
        "import torch, torchvision; m = torchvision.models.resnet18(); "
        "m.eval(); m(torch.randn(1, 3, 96, 96)); print('forward ok')"
    ),
}


def cpuinfo() -> None:
    try:
        with open("/proc/cpuinfo") as stream:
            for line in stream:
                low = line.lower()
                if low.startswith((
                    "model name", "model", "hardware", "features",
                    "cpu implementer", "cpu part", "cpu variant", "cpu architecture",
                )):
                    print(line.rstrip())
    except OSError:
        pass


def run_check(name: str, code: str) -> bool:
    print(f"\n== {name} ==", flush=True)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if proc.stdout.strip():
        print(proc.stdout.rstrip())
    if proc.returncode == 0:
        print("OK")
        return True
    if proc.returncode < 0:
        signal_number = -proc.returncode
        note = " (SIGILL)" if signal_number == 4 else ""
        print(f"FALHA: morto pelo sinal {signal_number}{note}")
    else:
        print(f"FALHA: retorno {proc.returncode}")
    if proc.stderr.strip():
        print(proc.stderr.rstrip())
    return False


def main() -> int:
    print(f"machine={platform.machine()} ({platform.platform()})")
    print(f"python={sys.version.split()[0]} em {sys.executable}")
    cpuinfo()
    results = {name: run_check(name, code) for name, code in CHECKS.items()}
    print("\n== resumo ==")
    for name, ok in results.items():
        print(f"{'OK   ' if ok else 'FALHA'} {name}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
