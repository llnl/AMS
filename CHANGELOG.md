# Changelog

## [Unreleased]

### Fixed

- Configured surrogate models now report load failures through catchable
  exceptions. Torch-disabled builds accept configured model paths, warn when
  an executor uses one, and run physics; warnings now appear by default.
- Invalid filesystem database paths now raise catchable exceptions while
  preserving the previous valid configuration for recovery.
- The RabbitMQ test consumer now creates non-durable AMS queues and replaces
  stale durable queues, preventing startup-order conflicts with AMS.
- Torch-enabled CMake package exports now resolve `AMS::TorchInterop`
  dependencies for downstream shared and static builds.
- Torch-disabled `AMSExecute` calls now preserve configured tensor and graph
  workflow storage when falling back to physics.

### Added

- JSON-backed storage now accepts AMSTensor directly, including strided and
  accelerator-resident tensors, and works in builds without Torch.
- JSON-backed storage can emit binary tensor files in rank-qualified case
  directories or self-contained base64 manifests, with a separate manifest for
  each domain and rank (#205).

### Changed

- `AMSGetAllocatorName` now returns an owning `std::string` instead of `const char*`, keeping returned names valid after allocator replacement; callers must rebuild and use the new return type.
- Workflow environments can now use active system Flux Python bindings instead
  of installing `flux-python` through default AMS Python dependencies.
- Added `AMS_INSTALL_FLUX_PYTHON` so non-system workflow builds can opt into
  installing the `flux-python` optional dependency through CMake.
