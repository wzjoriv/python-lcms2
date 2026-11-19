---
name: Bug report
about: A transform, profile or colormap gives wrong or unexpected results, or python-lcms2 fails
title: ''
labels: bug
assignees: wzjoriv

---

**Describe the bug**
What happened, and what you expected instead.

**Minimal example**
The smallest script that reproduces the problem:

```python
```

**Expected result**
What the example should print or return.

**Actual result / traceback**
What it prints or returns instead, including the full traceback if there is one.

```text
```

**ICC profiles involved**
Built-in profile name (for example `sRGB`, `Lab`), or attach the `.icc` / `.icm` file to the issue.

**Environment**
 - python-lcms2 version (`python -c "import lcms2; print(lcms2.__version__)"`):
 - LittleCMS version (`python -c "import lcms2; print(lcms2.get_version())"`):
 - NumPy version (`python -c "import numpy; print(numpy.__version__)"`):
 - Python version (`python --version`):
 - OS and architecture (for example Windows 11 x86_64, macOS 15 arm64, Ubuntu 24.04 aarch64):
 - Install method (wheel from PyPI, or source build; for a source build, say which commit):

**Additional context**
Anything else that helps: screenshots of the output, links, or what you already tried.
