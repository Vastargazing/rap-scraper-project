# VS Code Integration for lint.py

This guide shows how to integrate `lint.py` into VS Code for maximum productivity.

## 🚀 Quick Setup (5 minutes)

### Option 1: Tasks + Keyboard Shortcuts (Recommended)

**1. Add Tasks** to `.vscode/tasks.json`:

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Lint: Python Check",
      "type": "shell",
      "command": "python",
      "args": ["lint.py", "check"],
      "group": "build",
      "problemMatcher": [],
      "presentation": {
        "echo": true,
        "reveal": "always",
        "focus": false,
        "panel": "shared"
      }
    },
    {
      "label": "Lint: Python Fix",
      "type": "shell",
      "command": "python",
      "args": ["lint.py", "fix"],
      "group": "build",
      "problemMatcher": []
    },
    {
      "label": "Lint: Python All",
      "type": "shell",
      "command": "python",
      "args": ["lint.py", "all"],
      "group": {
        "kind": "build",
        "isDefault": true
      },
      "problemMatcher": []
    },
    {
      "label": "Lint: Python All (with logs)",
      "type": "shell",
      "command": "python",
      "args": ["lint.py", "all", "--log"],
      "group": "build",
      "problemMatcher": []
    }
  ]
}
```

**2. Add Keyboard Shortcuts** to `.vscode/keybindings.json`:

```json
[
  {
    "key": "ctrl+shift+l",
    "command": "workbench.action.tasks.runTask",
    "args": "Lint: Python Check",
    "when": "editorFocus"
  },
  {
    "key": "ctrl+shift+f",
    "command": "workbench.action.tasks.runTask",
    "args": "Lint: Python Fix",
    "when": "editorFocus"
  },
  {
    "key": "ctrl+shift+alt+l",
    "command": "workbench.action.tasks.runTask",
    "args": "Lint: Python All",
    "when": "editorFocus"
  }
]
```

**3. Usage:**

- **Ctrl+Shift+L**: Quick lint check
- **Ctrl+Shift+F**: Auto-fix issues
- **Ctrl+Shift+Alt+L**: Full pipeline (check + format + mypy)
- **Ctrl+Shift+P** → "Tasks: Run Task" → choose task

---

## 🎯 Why NOT a VS Code Extension?

### ❌ Extension Disadvantages:

| Aspect | Extension | Tasks + Keybindings |
|--------|-----------|---------------------|
| **Setup Time** | 8-12 hours | 5 minutes |
| **Maintenance** | Ongoing updates | None |
| **Complexity** | TypeScript, VS Code API | JSON config |
| **Distribution** | Marketplace account | Copy-paste |
| **Cross-project** | Need to install | Works immediately |

### ✅ Current Setup Advantages:

- **Fast**: 5 minutes setup vs 12 hours for extension
- **Simple**: Just JSON config files
- **Portable**: Copy `.vscode/` folder to any project
- **No dependencies**: Works with existing `lint.py`
- **Native**: Uses VS Code's built-in task system

### 🤔 When to Build Extension:

1. **Public sharing** - Want to publish on VS Code Marketplace
2. **Complex UI** - Need webviews, custom panels, decorations
3. **Learning project** - Want to learn VS Code extension API
4. **Advanced features** - Integration with VS Code language services

**For personal use:** Tasks + Keybindings is **better** (faster, simpler, maintainable).

---

## 📊 Feature Comparison

| Feature | Extension | Tasks |
|---------|-----------|-------|
| Run lint check | ✅ Button click | ✅ Ctrl+Shift+L |
| Auto-fix | ✅ Button click | ✅ Ctrl+Shift+F |
| Full pipeline | ✅ Button click | ✅ Ctrl+Shift+Alt+L |
| Command Palette | ✅ Yes | ✅ Yes |
| Keyboard shortcuts | ✅ Yes | ✅ Yes |
| Status bar integration | ✅ Yes | ❌ No |
| Custom UI panels | ✅ Yes | ❌ No |
| File watcher | ✅ Yes | ⚠️ Manual |
| Setup time | ❌ 12 hours | ✅ 5 minutes |
| Maintenance | ❌ High | ✅ None |

**Verdict:** For 90% of use cases, Tasks are sufficient and better.

---

## 🚀 Advanced: Task Runner Extension (Alternative)

Instead of building custom extension, use existing task runners:

### Option 1: Task Explorer Extension

```bash
# Install Task Explorer
code --install-extension spmeesseman.vscode-taskexplorer
```

- Adds **sidebar panel** with all tasks
- One-click task execution
- No custom code needed!

### Option 2: Tasks Shell Input Extension

```bash
# Install Tasks Shell Input
code --install-extension augustocdias.tasks-shell-input
```

- Run tasks with parameters
- Interactive prompts
- Extends built-in task system

---

## 🎯 Recommended Workflow

1. **Daily dev work:**
   - Press **Ctrl+Shift+L** to quick check
   - Save time, stay in flow

2. **Before commit:**
   - Press **Ctrl+Shift+Alt+L** for full pipeline
   - Catch all issues

3. **CI/CD:**
   ```bash
   python lint.py all --log
   ```
   - Full logs for debugging
   - History tracking

---

## 💡 Pro Tips

### 1. Make Default Build Task

In `tasks.json`, set `"isDefault": true`:

```json
{
  "label": "Lint: Python All",
  "group": {
    "kind": "build",
    "isDefault": true
  }
}
```

Now press **Ctrl+Shift+B** to run full lint!

### 2. Add to .vscode/settings.json

```json
{
  "task.autoDetect": "on",
  "task.quickOpen.history": 5,
  "task.quickOpen.showIcons": true
}
```

### 3. Pre-commit Hook Integration

Add to `.git/hooks/pre-commit`:

```bash
#!/bin/sh
python lint.py check
if [ $? -ne 0 ]; then
    echo "❌ Linting failed! Fix errors before commit."
    exit 1
fi
```

---

## 📚 Resources

- [VS Code Tasks Documentation](https://code.visualstudio.com/docs/editor/tasks)
- [Keybindings Documentation](https://code.visualstudio.com/docs/getstarted/keybindings)
- [Task Explorer Extension](https://marketplace.visualstudio.com/items?itemName=spmeesseman.vscode-taskexplorer)

---

**Status:** ✅ Production Ready  
**Last Updated:** October 2025  
**Recommendation:** Use Tasks + Keybindings (not extension)
