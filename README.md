The goal of this project is to be able to detect bib at the end of a race, in order to know each runners finish time.
Time measurement precision is discarded because any timestamp of the bib detected is sufficient enough for the finish time.

## Development

Requires [`uv`](https://docs.astral.sh/uv/). The Python version is pinned in
`.python-version`.

```powershell
uv sync                  # create/refresh the .venv
uv run pytest            # run the test suite
uv run ruff format .     # format the code
uv run ruff check .      # lint the code
uv run ty check          # type check the code
```

### Pre-commit hooks

File hygiene, formatting, linting, type checking and the test suite run
automatically before every commit. The hooks are defined in
`.pre-commit-config.yaml`.

```powershell
uv run pre-commit install          # install the hook, once per clone
uv run pre-commit run --all-files  # run every hook against every file
uv run pre-commit autoupdate       # bump the pinned hook revisions
```

Keep the `ruff` and `ty` versions in `pyproject.toml` in sync with the matching
`rev`s in `.pre-commit-config.yaml`.
