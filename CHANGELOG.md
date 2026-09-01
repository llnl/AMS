# Changelog

## [Unreleased]

### Added

- JSON-backed storage can emit binary tensor files or self-contained base64
  manifests named for each domain and rank.

### Changed

- Workflow environments can now use active system Flux Python bindings instead
  of installing `flux-python` through default AMS Python dependencies.
- Added `AMS_INSTALL_FLUX_PYTHON` so non-system workflow builds can opt into
  installing the `flux-python` optional dependency through CMake.
