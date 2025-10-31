# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.2] - 2025-10-07

Fixed bugs in the MACE-OMOL interface.

Fixed distributed training.

Added default spin and charge states in the atomic graph.
This allows using model's which use the above as input features to the model, e.g. MACE-OMOL.

Also added a MACE-OMOL convenience function.

Improved dev experience using pre-commit hooks.

Avoid directly overriding `numel` method for `PerElementParameter`s.

Fixed a bug where the threebody edge pairs were not being cached correctly.

# [0.1.6] - 2025-06-03

Added a ScaledHuberLoss

Added documentation for intergrating with SLURM

# [0.1.5] - 2025-06-03

Add learning rate warm up and HuberLoss options.

Small improvements to the GraphPESCalculator and OrbWrapper classes.

# [0.1.3] - 2025-06-02

Add explicit checking for unsupported elements in MACE interface.

## [0.1.2] - 2025-06-02

Extracted a base class for all interface models.

## [0.1.1] - 2025-05-01

Added the `egret` series of foundation models.

Simplified the configuration files.

## [0.1.0] - 2025-04-22

Added a `UnitConverter` model for converting between different unit systems.

Added native support for `torch-sim` integration, together with documentation and a tutorial.

## [0.0.36] - 2025-04-16

Added the `StillingerWeber` empirical potential.

## [0.0.35] - 2025-04-14

Improved quickstart documentation.

Added an interface to the `orb-models` package.

## [0.0.34] - 2025-04-10

Fix `graph-pes` dependency issues with `e3nn`.

## [0.0.32] - 2025-04-10

Added auto-offset fitting, together with documentation for how to fine-tune foundation models.

## [0.0.30] - 2025-04-03

Changed the way RMSE values are logged to ensure that they are correctly accumulated when validating (minor changes to final values).

## [0.0.29] - 2025-04-01

Add the `EDDP` architecture.

Add the `ConcatDataset` class.

## [0.0.26] - 2025-03-15

Fixed a bug whereby gradients were not being propagated through the three-body angle and distance terms.

Improved handling of early stopping.

Fixed a bug where custom callbacks were causing graph-pes-train to fail.

## [0.0.24] - 2025-02-15

Use [`vesin`](https://luthaf.fr/vesin/latest/index.html#) for accelerated neighbour list construction.

Add `ase_calculator` method to `GraphPESModel` for easy access to an ASE calculator wrapping the model.

Update the `mace` interfaces to use the default torch dtype if none is specified.

Add `ruff` check to CI.

## [0.0.22] - 2025-02-05

Add support for the `MatterSim` potential.

**Breaking change**: removed the `WeightedLoss` class, and placed the weight directly on the `Loss` instance.

Added a `summary.yaml` file to the output of each training/testing run that stores results locally.

## [0.0.21] - 2025-01-26

Fixed an inconsistency in the output shapes of `mace-torch` model predictions.

Added documentation for the `freeze` family of functions.

Fixed numerical instability in the `PaiNN` model.

## [0.0.19] - 2025-01-22

### Added

Added more fine-grained control over parameter freezing.

### Fixed

Parameter counting bug

## [0.0.18] - 2025-01-03

### Added

Added support for using arbitrary ``mace-torch`` models within ``graph-pes``, including the ``MACE-MP`` and ``MACE-OFF`` foundation models.

Support for custom batchers for properties in the ``other`` field via the `@register_custom_batcher` decorator.

### Changed

Updated the documentation for the `graph-pes-train` command.

## [0.0.17] - 2024-12-19

Add support for ASE `.db` files for backing datasets.

Simplified the logic for distributed training set ups.

Added the `graph-pes-test` command.

## [0.0.15] - 2024-12-13

### Changed

Changed the implementation of `TensorNet` to exactly match that in `TorchMD`.

### Fixed

Fixed test warnings.

Upgraded `locache` to `4.0.2` to fix some dataset caching issues.

Removed coloured logging for cleaner SLURM etc. output.

## [0.0.14] - 2024-12-12

### Added

Added the `graph-pes-resume` script.

Flushed logging for the `LoggedProgressBar` callback.

### Changed

Use explicit `.cutoff`, `.batch` and `.ptr` properties on `AtomicGraph` objects

Aligned the `graph-pes` implementation of MACE with that of `ACEsuite/mace`.

### Fixed

Fixed a bug whereby some parameters were being duplicated within a single parameter group.

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