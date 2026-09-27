The goal of this project is to create a SaaS application to be able to detect bib at the end of a race, in order to know each runners finish time.
This app could help race organizers to reduce waste and simplify times measurement. It also aims to improve organizers, runners and spectators experience through differents features.
Time measurement precision is discarded because any timestamp of the bib detected is sufficient enough for the finish time : since the video feed will likely film a runner during ~10sec, the finish time incertitude is around 10sec which is good enough for now.

## Selling Ideas

- For examples, we could add checkpoint camera that could help locate runners. This could be particularly useful for trail races where chip checkpoint can't be installed easily in the race route.
- The UI should improve the actual runners and spectator experience.
- Keep record of videos for manual review if needed.
- Personal finish clip
- “Last seen” spectator page

## Development

Requires [`uv`](https://docs.astral.sh/uv/). The Python version is pinned in
`.python-version`.

Always use uv to run python code.

```powershell
uv sync                  # create/refresh the .venv
uv run pytest            # run the test suite
uv run ruff format .     # format the code
uv run ruff check .      # lint the code
uv run ty check          # type check the code
uv add numpy             # Add numpy to dependencies
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
