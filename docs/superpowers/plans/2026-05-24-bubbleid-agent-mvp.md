# BubbleID Agent MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a separate BubbleID-Agent Python CLI that validates BubbleID projects, runs BubbleID reproducibly, inspects outputs, and drafts research reports.

**Architecture:** The repo is a thin orchestration layer around BubbleID. Core modules own config loading, project validation, BubbleID invocation, output inspection, report generation, and CLI wiring. The OpenAI report writer uses the Responses API when a key is available and provides a deterministic offline fallback for tests and non-API usage.

**Tech Stack:** Python 3.10+, argparse, dataclasses, pytest, optional BubbleID, optional OpenAI Python SDK.

---

### Task 1: Project Scaffolding And Tests

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `tests/test_config.py`
- Create: `tests/test_checks.py`
- Create: `tests/test_runner.py`
- Create: `tests/test_inspector.py`
- Create: `tests/test_reporter.py`

- [x] **Step 1: Write failing tests**

Tests cover JSON config parsing, warnings for missing calibration or low frame rate, manifest-only BubbleID runs, output anomaly detection, and offline report fallback.

- [x] **Step 2: Run tests to verify they fail**

Run: `python -m pytest`
Expected: fails because `bubbleid_agent` modules do not exist yet.

### Task 2: Core Modules

**Files:**
- Create: `src/bubbleid_agent/__init__.py`
- Create: `src/bubbleid_agent/config.py`
- Create: `src/bubbleid_agent/checks.py`
- Create: `src/bubbleid_agent/runner.py`
- Create: `src/bubbleid_agent/inspector.py`
- Create: `src/bubbleid_agent/reporter.py`
- Create: `src/bubbleid_agent/cli.py`

- [ ] **Step 1: Implement config dataclasses**

Load JSON config into `AnalysisConfig`, normalize paths, and expose manifest-safe dictionaries.

- [ ] **Step 2: Implement project checks**

Return structured issues with `error`, `warning`, or `info` severity.

- [ ] **Step 3: Implement BubbleID runner**

Write a manifest and optionally call `BubbleID.BubbleID.DataAnalysis.GenerateData`.

- [ ] **Step 4: Implement output inspector**

Scan expected output files and numeric CSV/TXT data for suspicious values.

- [ ] **Step 5: Implement report writer**

Use OpenAI Responses API when available; otherwise produce a deterministic Markdown report.

- [ ] **Step 6: Implement CLI**

Expose `check-project`, `run-analysis`, `inspect-outputs`, and `write-report`.

### Task 3: Documentation And Examples

**Files:**
- Create: `README.md`
- Create: `examples/config.example.json`

- [ ] **Step 1: Document purpose and install**

Explain that BubbleID-Agent is separate from BubbleID and depends on BubbleID for scientific CV analysis.

- [ ] **Step 2: Document commands**

Provide reproducible CLI examples for checking, running, inspecting, and reporting.

### Task 4: Verification And Publication

**Files:**
- Modify: repository git metadata

- [ ] **Step 1: Run tests**

Run: `python -m pytest`
Expected: all tests pass.

- [ ] **Step 2: Run CLI smoke checks**

Run: `python -m bubbleid_agent.cli --help`
Expected: command list is shown.

- [ ] **Step 3: Create GitHub repository**

Create `UARK-NED3/BubbleID-Agent` and push the initial branch.
