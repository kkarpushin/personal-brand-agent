---
name: windows-encoding
description: Ensures proper UTF-8 encoding in Python code for Windows compatibility. Use when writing file operations, subprocess calls, pathlib read/write, JSON/CSV handling, or any I/O in Python. Triggers on open(), read_text(), write_text(), subprocess.run(), json.dump().
---

# Windows Python Encoding

This project runs on Windows where Python defaults to cp1251/cp1252 encoding instead of UTF-8.
Always specify encoding explicitly to avoid UnicodeDecodeError and UnicodeEncodeError.

## Rules

### 1. File Operations - ALWAYS specify encoding

```python
# Reading files
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Writing files
with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
```

### 2. Pathlib - ALWAYS specify encoding

```python
from pathlib import Path

content = path.read_text(encoding='utf-8')
path.write_text(content, encoding='utf-8')
```

### 3. Subprocess - ALWAYS specify encoding and error handling

```python
import subprocess

result = subprocess.run(
    cmd,
    capture_output=True,
    text=True,
    encoding='utf-8',
    errors='replace'  # Replaces invalid chars instead of crashing
)
```

### 4. JSON Files

```python
import json

# Writing - use ensure_ascii=False for Cyrillic
with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

# Reading
with open('data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
```

### 5. CSV Files - Use utf-8-sig for Excel compatibility

```python
import csv

# utf-8-sig adds BOM for Excel to recognize UTF-8
with open('data.csv', 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerows(data)
```

### 6. Logging

```python
handler = logging.FileHandler('app.log', encoding='utf-8')
```

## Quick Reference

| Operation | Required Parameter |
|-----------|-------------------|
| `open()` | `encoding='utf-8'` |
| `Path.read_text()` | `encoding='utf-8'` |
| `Path.write_text()` | `encoding='utf-8'` |
| `subprocess.run()` | `encoding='utf-8', errors='replace'` |
| `json.dump()` | `ensure_ascii=False` + file `encoding='utf-8'` |
| CSV for Excel | `encoding='utf-8-sig'` |

## Common Mistakes

```python
# BAD - Will use cp1251 on Windows
with open(path, 'r') as f:
content = path.read_text()
subprocess.run(cmd, capture_output=True, text=True)

# GOOD - Explicit UTF-8
with open(path, 'r', encoding='utf-8') as f:
content = path.read_text(encoding='utf-8')
subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
```

