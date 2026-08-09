# Changelog

All notable changes to the TitanGPT Python SDK are documented in this file.

## [0.2.4] - 2026-08-10

### Added

- OpenAI-compatible error metadata on SDK exceptions: `error_type`, `param`,
  `code`, `request_id`, `status_code`, and `response_body`.
- A `py.typed` marker for type-checking tools.
- Automated sync and async error-handling tests.

### Changed

- Sync and async clients now use the same error parser and exception mapping.
- `model_not_found` responses are identified by the API error code, with a
  compatibility fallback for legacy responses.
- Package metadata, project links, and documentation were refreshed for PyPI.

