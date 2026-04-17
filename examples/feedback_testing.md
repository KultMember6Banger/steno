---
name: Integration Tests Must Hit Real DB
description: No mocking database in integration tests — learned from prod migration failure Q1
type: feedback
format: steno
---

# Integration tests must hit real database, not mocks

rule: integration tests connect to real PostgreSQL instance, never mock DB layer

**Why:** Q1 migration failure — mocked tests passed but prod migration broke because mock didn't enforce FK constraints. 3 hours of downtime.

**How to apply:**
  - `tests/integration/` → always uses `test_db` fixture (real PostgreSQL)
  - `tests/unit/` → mocks are fine here (testing logic, not persistence)
  - CI pipeline spins up PostgreSQL container per run
  - If test is slow because of real DB, optimize query or fixture, don't mock
