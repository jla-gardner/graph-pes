# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

Added the `graph-pes-resume` script.

### Changed

Use explicit `.cutoff`, `.batch` and `.ptr` properties on `AtomicGraph` objects

## [0.0.13] - 2024-12-11

### Added

Added verbose logging to the early training callback.

### Changed

[dev] moved ruff formatting to pre-commit hooks.

bumped `load-atoms` to 0.3.8.

## [0.0.12] - 2024-12-10

### Changed

Updated the minimum Python version to 3.9.

Un-pinned the `ase==3.22` dependency.

Migrate to using `data2objects>=0.1`

## [0.0.11] - 2024-12-08

### Added

Users can now freeze model components when they load them.

A new `ScalesLogger` callback is available to log per-element scaling factors.

A new, general `Loss` base class. The existing `Loss` class has now been renamed to `PropertyLoss`, and inherits from the new base class.

### Changed

Fix a bug in the `analysis.parity_plot` function.

Fixed a bug in the `OffsetLogger` callback.

## [0.0.10] - 2024-12-05

## [0.0.9] - 2024-12-05

### Added

Added neighbour-triplet based properties.

Generalised summations of properties defined on arbitrary collections of central atoms.

### Changed

Fix a bug when batching graphs with different properties.

Made training runs less verbose (redirected to `logs/rank-0.log`).

## [0.0.8] - 2024-12-04

### Added

Support for `"virial"` property predictions, as well as `"stress"`.

### Changed

Migrated to using [data2objects](https://github.com/jla-gardner/data2objects) for configurations - this affects all configuration files.

Improved saving behaviour of models.

Improved the documentation for the `PerElementParameter` class.

## [0.0.7] - 2024-12-02

### Added

Allow the user to freeze parameters in models that they load (useful for e.g. fine-tuning).

## [0.0.6] - 2024-11-29

### Changed

Fix a bug where the model was not being saved correctly.

## [0.0.5] - 2024-11-29

### Added

Allow for using arbitrary devices with `GraphPESCalculator`s.

Allow the user to configure and define custom callbacks for the trainer. Implemented `OffsetLogger` and `DumpModel`.

## [0.0.4] - 2024-11-26

### Added

Automatically detect validation metrics from available properties.

### Changed

Improved documentation for LAMMPS integration.

Fixed a bug where stress was not converted to 3x3 from Voigt notation in some cases.

## [0.0.3] - 2024-10-31