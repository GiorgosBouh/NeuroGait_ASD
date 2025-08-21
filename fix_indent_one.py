#!/usr/bin/env python3
import pathlib, subprocess, ast, shutil

p = pathlib.Path("neurogait_fixed.py").resolve()
bak = p.with_suffix(".py.bak")

# 1) backup
shutil.copy2(p, bak)

# 2) untabify -> 4 spaces
txt = p.read_text(encoding="utf-8")
txt2 = txt.replace("\t", "    ")
if txt2 != txt:
    p.write_text(txt2, encoding="utf-8")
    print("[UNTABIFY] tabs -> 4 spaces")

# 3) optional: run black if installed
try:
    subprocess.run(["black", str(p)], check=False)
except Exception:
    print("[INFO] black not found, skipping format")

# 4) syntax check
try:
    ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
    print("[SYNTAX] OK")
except SyntaxError as e:
    print(f"[SYNTAX] ERROR: {e}")

# 5) tabnanny (indentation issues)
subprocess.run(["python", "-m", "tabnanny", "-v", str(p)])
print("Done.")