# 🔥 Linting Setup - Quick Reference

**Project:** Rap Scraper/Analyzer
**Linter:** Ruff 0.14.1
**Type Checker:** Mypy 1.18.2
**Status:** ✅ Fully Configured

---

# 🔥 Linting Setup - Quick Reference

**Project:** Rap Scraper/Analyzer
**Linter:** Ruff 0.14.1
**Type Checker:** Mypy 1.18.2
**Status:** ✅ Fully Configured

---

## 🚀 Quick Start (Modern Python Way - **RECOMMENDED**)

### **Best Practice: Use `lint.py` Python Script** 🐍

✅ **Cross-platform** (Windows/Linux/Mac)
✅ **No encoding issues** (handles emoji/Unicode properly)
✅ **Simple** (just Python, no PowerShell/Bash complexity)

```bash
# Check for issues (fast dev loop)
python lint.py check

# Auto-fix issues
python lint.py fix

# Full pipeline (check + format + type check)
python lint.py all

# With file logging (CI/CD, history tracking)
python lint.py all --log

# Watch mode (auto-lint on file changes - requires watchdog)
python lint.py watch
```

---

## 📜 Alternative Methods (Legacy)

### Option 1: Direct Ruff Commands

```bash
# Check for issues
ruff check .

### Option 2: Direct Ruff/Mypy Commands (If you prefer manual control)

```bash
# Check
ruff check .

# Fix
ruff check --fix .

# Format
ruff format .

# Type check
mypy .

# All together
ruff check . && ruff format . && mypy .
```

### Option 3: PowerShell Script (Windows - Legacy, NOT RECOMMENDED)

**⚠️ Note:** PowerShell scripts have emoji/encoding issues on Windows. Use `lint.py` instead!

```powershell
# Check
./lint.ps1 -Command check

# Fix and format
./lint.ps1 -Command fix

# Full pipeline
./lint.ps1 -Command all

# Watch for changes
./lint.ps1 -Command watch
```

### Option 4: Make/Unix (Linux/Mac)

```bash
# Check
make lint

# Fix and format
make format && make check

# CI full pipeline
make ci-lint
```

### Option 5: VS Code Tasks

Press **Ctrl+Shift+P** → `Tasks: Run Task` and choose:
- **Lint: Ruff Check** - Check without fixes
- **Lint: Ruff Fix** - Automatically fix
- **Lint: Ruff Format** - Format code
- **Lint: Mypy Check** - Type checking
- **Lint: All (Ruff + Mypy)** - Full check

### Option 6: VS Code Keyboard Shortcuts

| Shortcut | Action |
|---|---|
| **Alt+Shift+F** | Format current file |
| **Ctrl+Shift+M** | Open Problems panel |
| **F8** | Next problem |
| **Shift+F8** | Previous problem |

---

## 📋 Configuration Files

### `.ruff.toml` - Ruff Configuration
- **Target Python:** 3.10+
- **Line length:** 88 characters
- **Quote style:** Double quotes
- **Enables:** 25+ linting categories

### `pyproject.toml` - Mypy Configuration
- **Mode:** Strict type checking
- **Disallows:** Untyped definitions, incomplete defs
- **Type stubs:** Required for all functions

### `.pre-commit-config.yaml` - Git Pre-commit Hooks
- Auto-lint before every commit
- Install: `pip install pre-commit && pre-commit install`
- Test: `pre-commit run --all-files`

---

## 🔍 What Gets Checked

### Ruff (Linting & Formatting)
- ✅ Code style (PEP 8)
- ✅ Imports (sorting, unused)
- ✅ Naming conventions (PEP 8)
- ✅ Potential bugs (mutable defaults, etc.)
- ✅ Code simplification
- ✅ Performance issues
- ✅ Docstring formatting
- ✅ And 18+ more categories

### Mypy (Type Checking)
- ✅ Type annotations present
- ✅ Type compatibility
- ✅ Return type correctness
- ✅ Optional/None handling
- ✅ Generic types (List, Dict, etc.)

---

## 📊 Example: Before & After

### ❌ BEFORE (Fails linting)
```python
# Missing types, bad formatting, unused import
import os
def process(data):
    x=1+2;y=3
    return f"Result: {x}"
```

### ✅ AFTER (Passes linting)
```python
# Proper formatting, type hints, organized
def process(data: str) -> str:
    x = 1 + 2
    y = 3
    return f"Result: {x}"
```

---

## 🛠️ File Scripts

### `lint.ps1` (PowerShell - Windows)
```powershell
./lint.ps1 -Command check    # Check
./lint.ps1 -Command fix      # Fix
./lint.ps1 -Command format   # Format
./lint.ps1 -Command mypy     # Type check
./lint.ps1 -Command all      # Everything
./lint.ps1 -Command watch    # Watch mode
```

### `lint.sh` (Bash - Linux/Mac)
```bash
./lint.sh check    # Check
./lint.sh fix      # Fix
./lint.sh format   # Format
./lint.sh mypy     # Type check
./lint.sh all      # Everything
./lint.sh watch    # Watch mode
```

---

## 🎯 Typical Workflow (Modern Python Way)

### 1. Make changes to your code

```python
def analyze_rap_lyrics(lyrics):
    return lyrics.upper()
```

### 2. Quick check while developing

```bash
python lint.py check
```

### 3. Auto-fix issues before commit

```bash
python lint.py fix
```

### 4. Full check before commit (with logs for history)

```bash
python lint.py all --log
```

### 5. Commit (pre-commit hooks will check automatically)

```bash
git add .
git commit -m "feat: add lyrics analysis"
```

---

## 🆚 Tool Comparison

| Method | Platform | Emoji/UTF-8 | Recommended |
|--------|----------|-------------|-------------|
| **`lint.py`** | ✅ All (Win/Linux/Mac) | ✅ Perfect | ⭐ **YES** |
| `lint.ps1` | ❌ Windows only | ⚠️ Issues | ❌ NO |
| `lint.sh` | ❌ Linux/Mac only | ✅ Good | ❌ NO |
| Direct `ruff`/`mypy` | ✅ All | ✅ Good | ⚠️ Manual |
| VS Code Tasks | ✅ All | ✅ Good | ✅ OK |

**Winner: `python lint.py` - One tool to rule them all!** 🏆

---

## 🎯 Typical Workflow

### 1. Make changes to your code

```python
def analyze_rap_lyrics(lyrics):
    return lyrics.upper()
```

### 2. Save file (Ctrl+S)
- ✅ Ruff automatically formats
- ✅ Imports organized
- ✅ Issues highlighted

### 3. Before commit, check once more

```powershell
./lint.ps1 -Command all
# or
make check
```

### 4. Fix any remaining issues

```powershell
./lint.ps1 -Command fix
```

### 5. Commit (pre-commit hooks will check automatically)

```bash
git add .
git commit -m "feat: add lyrics analysis"
```

---

## 🚨 Common Issues & Solutions

### "ruff: command not found"
```powershell
# Make sure you're using venv Python
.\.venv\Scripts\ruff --version

# Or run from virtual environment
poetry shell
ruff --version
```

### "Mypy found errors in strict mode"
This is normal! Fix them gradually:
1. Start with **errors** (red)
2. Then **warnings** (yellow)
3. Then **info** (blue)

### "Too many linting errors?"
Start with just Ruff, then add Mypy:
```powershell
# First: Just format
ruff format .

# Then: Fix errors
ruff check --fix .

# Finally: Type checking
mypy .
```

---

## 📚 Files Created/Modified

✅ **Created:**
- `lint.ps1` - PowerShell script for linting
- `lint.sh` - Bash script for linting
- `.pre-commit-config.yaml` - Pre-commit hooks config
- `.github/workflows/lint.yml` - GitHub Actions CI

✅ **Modified:**
- `.vscode/settings.json` - Added Ruff + Mypy config
- `Makefile` - Updated lint commands to use Ruff/Mypy
- `pyproject.toml` - Already has mypy config
- `.ruff.toml` - Already configured (25+ rules)

---

## 🔗 Quick Links

- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Mypy Documentation](https://mypy.readthedocs.io/)
- [PEP 8 Style Guide](https://pep8.org/)
- [Python Type Hints](https://www.python.org/dev/peps/pep-0484/)

---

## ✨ Next Steps

1. **Install extensions** in VS Code if not done:
   ```
   Ctrl+Shift+P → Extensions: Install Extensions
   Search: "Ruff" and "Mypy"
   ```

2. **Reload VS Code**:
   ```
   Ctrl+Shift+P → Developer: Reload Window
   ```

3. **Test it works**:
   ```powershell
   ./lint.ps1 -Command all
   ```

4. **Configure pre-commit hooks** (optional):
   ```bash
   pip install pre-commit
   pre-commit install
   pre-commit run --all-files
   ```

---

**Status: ✅ Production Ready**
*Last Updated: October 2025*
