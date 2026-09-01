# Design Document

## Purpose
Provide a lightweight end-to-end task workspace that demonstrates a
modern full-stack workflow without unnecessary infrastructure.

## Architecture
Browser UI → Express REST API → SQLite

The server also exposes Server-Sent Events so connected clients receive
task updates without repeatedly refreshing the page.

## Main features
- Task creation
- Completion status
- Deletion
- Live updates
- Simple productivity totals

## Trade-off
This is a compact replication rather than a full clone of every feature
in the reference system. The professor's prompt allows reasonable
assumptions and creative implementation.
