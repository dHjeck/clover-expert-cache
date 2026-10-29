"""Measure direct SSD reads of Clover BF16 routed experts on Windows."""
import ctypes
import ctypes.wintypes as wt
import json
import mmap
import os
import random
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CLOVER_MODEL")
SAMPLE_COUNT = int(sys.argv[2]) if len(sys.argv) > 2 else 96
if not ROOT:
    raise SystemExit("Usage: python clover_ssd_bench.py <model-dir> [sample-count], or set CLOVER_MODEL")

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.CreateFileW.restype = wt.HANDLE
k32.CreateFileW.argtypes = [wt.LPCWSTR, wt.DWORD, wt.DWORD, ctypes.c_void_p, wt.DWORD, wt.DWORD, wt.HANDLE]
k32.ReadFile.argtypes = [wt.HANDLE, ctypes.c_void_p, wt.DWORD, ctypes.POINTER(wt.DWORD), ctypes.c_void_p]
k32.SetFilePointerEx.argtypes = [wt.HANDLE, ctypes.c_longlong, ctypes.c_void_p, wt.DWORD]
k32.CloseHandle.argtypes = [wt.HANDLE]

GENERIC_READ = 0x80000000
FILE_SHARE_READ = 0x00000001
OPEN_EXISTING = 3
FILE_FLAG_NO_BUFFERING = 0x20000000
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value


def index_experts(root):
    entries = []
    for fn in sorted(os.listdir(root)):
        if not fn.endswith(".safetensors"):
            continue
        path = os.path.join(root, fn)
        with open(path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
        for key, meta in header.items():
            if not key.startswith("model.layers.") or ".mlp.experts." not in key:
                continue
            parts = key.split(".")
            layer = int(parts[2])
            tensor = parts[-1]
            start, end = meta["data_offsets"]
            entries.append((layer, tensor, path, 8 + header_len + start, end - start))
    by_expert = {}
    for layer, tensor, path, offset, size in entries:
        for expert_id in range(256):
            by_expert.setdefault((layer, expert_id), []).append((path, offset + expert_id * (size // 256), size // 256))
    out = []
    for key, ranges in sorted(by_expert.items()):
        if len(ranges) != 2:
            raise RuntimeError(f"incomplete expert {key}: {ranges}")
        if sum(r[2] for r in ranges) != 6 * 2**20:
            raise RuntimeError(f"unexpected size for {key}: {sum(r[2] for r in ranges)}")
        out.append((key, ranges))
    return out


def read_range(path, offset, length):
    start = offset & ~4095
    end = (offset + length + 4095) & ~4095
    size = end - start
    handle = k32.CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, None, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, None)
    if handle in (None, 0, INVALID_HANDLE_VALUE):
        raise OSError(ctypes.get_last_error(), f"CreateFileW failed: {path}")
    buf = mmap.mmap(-1, size)
    try:
        addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
        if k32.SetFilePointerEx(handle, start, None, 0) == 0:
            raise OSError(ctypes.get_last_error(), "SetFilePointerEx failed")
        got = wt.DWORD()
        ok = k32.ReadFile(handle, addr, size, ctypes.byref(got), None)
        if not ok or got.value != size:
            raise OSError(ctypes.get_last_error(), f"ReadFile failed, got {got.value}/{size}")
    finally:
        del addr
        buf.close()
        k32.CloseHandle(handle)


def read_expert(entry):
    _, ranges = entry
    for path, offset, length in ranges:
        read_range(path, offset, length)
    return 6 * 2**20


def median(values):
    values = sorted(values)
    return values[len(values) // 2]


def main():
    experts = index_experts(ROOT)
    random.seed(20261028)
    sample = random.sample(experts, min(SAMPLE_COUNT, len(experts)))
    rows = []
    latencies = []
    for entry in sample:
        t0 = time.perf_counter()
        read_expert(entry)
        latencies.append(time.perf_counter() - t0)
    rows.append({
        "mode": "one_at_a_time",
        "experts": len(sample),
        "median_ms": median(latencies) * 1000,
        "p90_ms": sorted(latencies)[int(len(latencies) * 0.9)] * 1000,
        "mean_ms": sum(latencies) / len(latencies) * 1000,
        "throughput_gib_s": (len(sample) * 6 / 1024) / sum(latencies),
    })
    for workers in (2, 4, 8, 16):
        batch = random.sample(experts, min(SAMPLE_COUNT, len(experts)))
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            total = sum(pool.map(read_expert, batch))
        dt = time.perf_counter() - t0
        rows.append({
            "mode": f"{workers}_in_flight",
            "workers": workers,
            "experts": len(batch),
            "elapsed_s": dt,
            "throughput_gib_s": total / 2**30 / dt,
            "ms_per_expert": dt / len(batch) * 1000,
        })
    out = {
        "root": ROOT,
        "expert_count": len(experts),
        "expert_bytes": 6 * 2**20,
        "sample_count": len(sample),
        "direct_io": True,
        "rows": rows,
        "note": "FILE_FLAG_NO_BUFFERING reads the two gate_up_proj/down_proj ranges of each BF16 expert; results are storage-path measurements, not model throughput.",
    }
    out_path = os.path.join(os.path.dirname(__file__), "results", "clover-ssd-results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
