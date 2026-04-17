---
name: Production Stack Reference
description: Current production infrastructure and service map
type: reference
format: steno
---

# Production Stack

cloud: AWS (us-east-1 primary, eu-west-1 DR)
orchestration: EKS 1.29, Helm charts in infra/ repo
db: PostgreSQL 16 (RDS), Redis 7 (ElastiCache)
ci: GitHub Actions → ECR → ArgoCD
monitoring: Datadog APM + Grafana (grafana.internal/d/overview)
secrets: AWS Secrets Manager, rotated quarterly

services:
  user-service: Go 1.22, gRPC, port 50051
  auth-service: Go 1.22, gRPC, port 50052
  billing-service: Python 3.12, REST (migrating to gRPC)
  notification-service: Python 3.12, REST, SQS consumer
  web-frontend: Next.js 14, Vercel

oncall: PagerDuty rotation, runbooks at runbooks.internal/
