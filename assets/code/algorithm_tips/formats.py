"""What fp32, fp16, bf16 and int8 can and cannot store, checked with PyTorch on CPU."""
import importlib.metadata as md, sys
import torch

def bits(x, dtype):
    t = torch.tensor([x], dtype=dtype)
    width = {torch.float32: 32, torch.float16: 16, torch.bfloat16: 16}[dtype]
    raw = t.view({32: torch.int32, 16: torch.int16}[width]).item() & ((1 << width) - 1)
    s = f"{raw:0{width}b}"
    exp = {torch.float32: 8, torch.float16: 5, torch.bfloat16: 8}[dtype]
    return f"{s[0]} {s[1:1 + exp]} {s[1 + exp:]}"

print("1. Layout (sign | exponent | fraction) and limits")
for name, dt in (("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)):
    fi = torch.finfo(dt)
    print(f"  {name}: {fi.bits} bits, max {fi.max:.4g}, smallest normal {fi.tiny:.3g}, eps {fi.eps:.3g} | 0.1 = {bits(0.1, dt)}")

print("2. Range: 70,000 and 1e-8")
for name, dt in (("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)):
    print(f"  {name}: 70000 -> {torch.tensor(70000.0, dtype=dt).item():g}, 1e-8 -> {torch.tensor(1e-8, dtype=dt).item():g}")

print("3. Precision: 1 + 0.001, and adding 0.001 ten thousand times (exact answer 10)")
for name, dt in (("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)):
    one = torch.tensor(1.0, dtype=dt); step = torch.tensor(0.001, dtype=dt)
    total = torch.tensor(0.0, dtype=dt)
    for _ in range(10_000):
        total = total + step
    print(f"  {name}: 1 + 0.001 = {(one + step).item():.6f} | running sum = {total.item():g}")

print("4. int8: 8-bit two's complement, -128..127; weights need a scale")
torch.manual_seed(0)
w = torch.randn(4096) * 0.02                                  # typical small weights
def int8_roundtrip(x):
    scale = x.abs().max() / 127
    q = torch.clamp(torch.round(x / scale), -128, 127).to(torch.int8)
    return q, q.float() * scale, scale
for label, x in (("normal weights", w), ("same + one outlier of 1.0", torch.cat([w[:-1], torch.tensor([1.0])]))):
    q, back, scale = int8_roundtrip(x)
    err = (back[:4095] - x[:4095]).abs().mean() / x[:4095].abs().mean()
    print(f"  {label:26}: scale {scale.item():.6f}, distinct int8 values used {q.unique().numel():3d}, mean relative error {err.item():.1%}")
print("  bf16 of the normal weights: mean relative error",
      f"{((w.bfloat16().float() - w).abs().mean() / w.abs().mean()).item():.2%}")

print("5. Memory for 494 million parameters (Qwen2.5-0.5B)")
params = 494_032_768
for name, b in (("fp32", 4), ("fp16 / bf16", 2), ("int8", 1), ("int4", 0.5)):
    print(f"  {name:12}: {params * b / 2**30:.2f} GiB")
print("Python", sys.version.split()[0], "| torch", torch.__version__)
