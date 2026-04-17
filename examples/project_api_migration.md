---
name: API v2 Migration
description: REST to gRPC migration for internal services, deadline 2026-05-15
type: project
format: steno
---

# API v2 Migration

status: in progress
deadline: 2026-05-15
owner: platform team
driver: latency requirements from ML pipeline (p99 > 200ms on REST, target < 50ms)

completed:
  user-service: migrated 04-01, running in prod, 12ms p99
  auth-service: migrated 04-08, canary at 10%

in-progress:
  billing-service: proto files done, handler migration 60%, blocked on decimal precision in protobuf
  notification-service: not started, low priority (async, latency insensitive)

**Why:** ML pipeline calls user-service 50K/min. REST overhead dominates latency budget. gRPC binary framing + HTTP/2 multiplexing cuts p99 from 210ms to 12ms (measured on user-service).

**How to apply:** when touching any internal service API, check if it's pre or post migration. Pre-migration services still use REST clients. Don't add new REST endpoints to migrated services.
