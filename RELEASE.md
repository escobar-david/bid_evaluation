# Release Checklist (PyPI)

## 1) Pre-release checks

```bash
python -m pip install -e ".[dev]"
pytest -q
```

Confirm version is updated in:
- `pyproject.toml` (`[project].version`)
- `bid_evaluation/__init__.py` (`__version__`)

## 2) Build distributions

```bash
python -m build
```

Expected artifacts in `dist/`:
- `bid_evaluation-<version>-py3-none-any.whl`
- `bid_evaluation-<version>.tar.gz`

## 3) Validate package metadata

```bash
twine check dist/*
```

## 4) Test install from local artifacts (recommended)

```bash
python -m pip install --force-reinstall dist/bid_evaluation-<version>-py3-none-any.whl
python -c "import bid_evaluation; print(bid_evaluation.__version__)"
```

## 5) Upload to TestPyPI (recommended first)

```bash
twine upload --repository testpypi dist/*
```

Test install:

```bash
python -m pip install --index-url https://test.pypi.org/simple/ bid-evaluation
```

## 6) Upload to PyPI

```bash
twine upload dist/*
```

## 7) Post-release

- Create a Git tag (for example `v0.1.0`)
- Publish GitHub release notes
- Bump to next development version if needed
