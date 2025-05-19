# CI/CD Configuration

This directory contains continuous integration and continuous deployment configurations for AIQToolkit.

## Structure

- `release/` - Release management scripts and templates
  - `pr_code_freeze_template.md` - PR template for code freeze
  - `update-version.sh` - Version update script
  - `update_toml_dep.py` - TOML dependency updater

- `scripts/` - CI/CD automation scripts
  - `github/` - GitHub Actions specific scripts
  - `gitlab/` - GitLab CI specific scripts
  - `bootstrap_local_ci.sh` - Local CI environment setup
  - `checks.sh` - Code quality checks
  - `documentation_checks.sh` - Documentation validation
  - `python_checks.sh` - Python code validation
  - `run_ci_local.sh` - Run CI pipeline locally

- `vale/` - Documentation style checking
  - `styles/config/` - Vale style configurations

## Usage

Run CI checks locally:
```bash
./ci/scripts/run_ci_local.sh
```

Bootstrap local CI environment:
```bash
./ci/scripts/bootstrap_local_ci.sh
```