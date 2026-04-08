# Board Perception Integration

## Runtime Invocation Policy

- run during bootstrap
- run again only when Discovery Agent requests it
- do not run every round by default

## Consumers

- Discovery Agent consumes and stores the report
- Planner Agent may consume board output in v1
- Orchestrator remains authority
- module remains advisory-only

