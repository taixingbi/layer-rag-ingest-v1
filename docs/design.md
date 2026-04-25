# Design

## Overview

This document captures the system design for the RAG ingest pipeline.

## Goals

- Define the ingest architecture and data flow.
- Describe key components and responsibilities.
- Document operational considerations and trade-offs.

## Scope

- In scope: data ingestion, chunking, payload preparation, synthetic question enrichment, embedding, and Qdrant upsert.
- Out of scope: downstream retrieval and application-layer response orchestration.

## High-Level Flow

1. Read raw source files.
2. Create chunk JSON outputs.
3. Prepare metadata-rich point payloads.
4. Optionally enrich with synthetic questions.
5. Embed and upsert points to Qdrant.

## Components

- `app/plain_text_chunks.py`
- `app/markdown_to_chunks.py`
- `app/prepare_payloads.py`
- `app/synthetic_questions.py`
- `app/upsert_qdrant.py`

## Notes

Use this file as the canonical design reference and keep it aligned with implementation changes.
