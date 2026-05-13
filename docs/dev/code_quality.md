# Code Quality

All linting and formatting checks are defined in `.pre-commit-config.yaml` and
run automatically when you commit. Once set up, you never need to invoke a
linter manually.

## Setup

Install the Git hooks (one-time):

```bash
uv run prek install
```

After this, every `git commit` runs the configured hooks. If a hook fails, the
commit is blocked until you fix the issue.

## What gets checked

| Tool / Hook                                | Purpose                                             |
| ------------------------------------------ | --------------------------------------------------- |
| `ruff-check`                               | Python linting and style enforcement                |
| `ruff-format`                              | Python code formatting                              |
| `typos`                                    | Spell checking in source and docs                   |
| `prettier`                                 | Markdown / YAML / JSON formatting                   |
| `markdownlint-cli2`                        | Markdown style rules                                |
| `check-added-large-files`                  | Prevents accidentally committing large files        |
| `check-case-conflict`                      | Detects case-conflicting filenames                  |
| `check-merge-conflict`                     | Detects leftover conflict markers                   |
| `check-symlinks`                           | Detects broken symlinks                             |
| `check-yaml` / `check-toml` / `check-json` | Validates structured file syntax                    |
| `debug-statements`                         | Catches leftover `breakpoint()` / `pdb.set_trace()` |
| `end-of-file-fixer`                        | Ensures files end with a newline                    |
| `mixed-line-ending`                        | Normalises line endings                             |
| `trailing-whitespace`                      | Removes trailing whitespace                         |
| `check-docstring-first`                    | Ensures docstrings come before other code           |
| `uv-lock`                                  | Keeps `uv.lock` in sync with `pyproject.toml`       |
| `gitleaks`                                 | Detects secrets and credentials                     |

## Skipping hooks (emergency only)

```bash
git commit --no-verify -m "message"
```

Use sparingly — CI will still enforce the same checks on pull requests.

## CI

The same checks run in GitHub Actions on every pull request and push to the
default branch, so even if hooks are bypassed locally, the CI pipeline catches
issues.
