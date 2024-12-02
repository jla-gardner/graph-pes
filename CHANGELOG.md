# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

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