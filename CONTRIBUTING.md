# Contributing

Bug reports and pull requests are welcome at
<https://github.com/tysonjohnsondev/netviz_tools>.

## Setup

The project uses [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/tysonjohnsondev/netviz_tools
cd netviz_tools
uv sync
uv run pre-commit install
```

## Checks

Pull requests must pass the same checks as CI:

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy                    # strict mode
uv run pytest                  # includes doctests; fails below 90% coverage
uv run --extra docs mkdocs build --strict
```

Tests that download real data are marked `network` and skipped by default. Run
them with `uv run pytest -m network`.

## Guidelines

- Functions return values. They do not print, log, show figures, or write files
  unless writing files is their purpose.
- Public functions have type hints and numpy-style docstrings.
- Keep the flow-frame schema as the common input.
- Add a line under "Unreleased" in `CHANGELOG.md` for user-facing changes.

## Regenerating bundled data

- `scripts/build_metadata.py` rebuilds the country and item tables from FAOSTAT,
  UNSD M49 and Natural Earth, and records sources and overrides in
  `sources.json`.
- `scripts/make_sample.py` cuts the sample Parquet file from a local store.
- `scripts/make_hero.py` renders the README image
  (`uv run --with kaleido python scripts/make_hero.py`).

## Releasing

1. Update the version in `pyproject.toml` and move the "Unreleased" notes in
   `CHANGELOG.md` under the new version.
2. Commit, then tag: `git tag v1.2.3 && git push --tags`.
3. The release workflow checks that the tag matches the version, runs the
   tests, builds, and publishes to PyPI with Trusted Publishing.
