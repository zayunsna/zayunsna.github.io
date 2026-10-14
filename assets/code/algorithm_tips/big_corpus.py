"""Larger real-text corpus without imports: docstring paragraphs parsed with ast from every .py file
in the standard library and this environment's site-packages."""
import ast, json, sys, sysconfig
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

roots = [Path(sysconfig.get_paths()["stdlib"]), Path(sysconfig.get_paths()["purelib"])]
texts = set()
for root in roots:
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                for para in (ast.get_docstring(node) or "").split("\n\n"):
                    s = " ".join(para.split())
                    if 40 <= len(s) <= 400 and not s.startswith((">>>", "..", "-", "=")):
                        texts.add(s)
texts = sorted(texts)
print(len(texts), "paragraphs")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
emb = model.encode(texts, batch_size=512, normalize_embeddings=True).astype(np.float32)
np.save("big.npy", emb); json.dump(texts, open("big.json", "w"))
print(emb.shape)
