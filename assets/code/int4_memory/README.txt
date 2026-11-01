Bounded memory accounting experiment (tested 2026-11-01)

Python 3.12.13, macOS Apple M5, CPU; no API key.
Install requirements.txt in a new virtual environment, then:
  python memory_parts.py --output memory.json
Add --local-files-only if the pinned Qwen checkpoint is already cached.
Model download is required otherwise. Model weights are not redistributed here.
Qwen2.5-0.5B-Instruct is Apache-2.0 licensed (official model card).

Two SEPARATE experiments, not a full int4 LLM inference benchmark:
1. Educational NumPy symmetric group-64 nibble packing and FP32 restoration.
2. Real Qwen base-transformer FP16 CPU cache payload at three bounded shapes.
The 3 GiB long-context cache is a formula projection, not an allocation.
Tensor/array payload != peak RSS. No intentional OOM test is performed.
Saved memory.json contains exact measured payloads and library versions.

To regenerate cover.png, install matplotlib==3.11.0 and run:
  python render_cover.py --output cover.png
It reads memory.json next to this file. Only measured bounded payloads plotted.
