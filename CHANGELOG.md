# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-06-16

### Added

- Initial package skeleton (src-layout, import name `lendingclub_default`).
- Core helpers: `_constants` (status vocabularies, grades, terms), `_typing`,
  `_exceptions` (`LendingClubDefaultError` base + `LeakageError`,
  `TemporalSplitError`, `ArtifactError`), `_validation`, `_manifest`
  (`RunManifest` with BLAKE2b config-hash), and `_rng` (seeded PCG64).
- `data/leakage.py` with the frozen, reviewed `LEAKAGE_COLS` post-funding
  allowlist (the canonical reference list) plus `drop_leakage` / `assert_no_leakage`.
- Typed stub signatures with full contracts for `data` (`synthetic`, `load`,
  `labels`, `split`), `features` (`pipeline`), `models` (`baselines`, `xgb`,
  `calibrate`, `reason_codes`), and `evaluation` (`metrics`, `calibration`,
  `delong`, `threshold`) subpackages.
- End-to-end `train`, Plotly figure builders (`plots`), and a Typer CLI (`cli`)
  stub.
- Partitioned test suite (unit / parity / property / regression / integration)
  and seeded conftest fixtures (`synthetic_panel`, `k_vintage_fixture`,
  `schema_with_leakage`).

[Unreleased]: https://github.com/FatihHekim0glu/lendingclub-default/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/FatihHekim0glu/lendingclub-default/releases/tag/v0.1.0
