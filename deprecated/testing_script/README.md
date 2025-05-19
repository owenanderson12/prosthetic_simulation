# Deprecated: `testing_script` Directory

## Purpose

The `testing_script` directory originally contained standalone scripts and utilities for testing individual components of the EEG-controlled prosthetic hand system. These scripts were used to:
- Validate early-stage data acquisition and processing pipelines
- Simulate and debug artificial EEG data sources
- Perform isolated unit and integration tests outside the main BCI workflow

## Deprecation Rationale

With the integration of the unified `DataSource` abstraction and improved modularity across the codebase, all core testing, calibration, and validation workflows are now handled within the main system (`main.py`) and its modules. The artificial data source and calibration can be fully tested end-to-end using the main command-line interface, making these legacy scripts redundant.

**This folder is now deprecated and retained only for historical reference.**
- All new tests and validation should be performed using the main BCI system or through modernized test suites.
- No further updates will be made to the scripts in this directory.

## Recommendation

Refer to the main project documentation and integrated test flows for current testing procedures. If you need to revisit legacy test logic, consult these scripts as reference only.
