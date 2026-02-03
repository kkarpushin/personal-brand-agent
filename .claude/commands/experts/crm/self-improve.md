---
allowed-tools: Read, Grep, Glob, Bash, Edit, Write, TodoWrite
description: Self-improve CRM expertise by validating against codebase
argument-hint: [check_git_diff (true/false)] [focus_area (optional)]
---

# CRM Expert - Self-Improve Mode

Maintain expertise accuracy by comparing against actual codebase implementation.

## Variables

- CHECK_GIT_DIFF: $1 (default: false)
- FOCUS_AREA: $2 (default: empty - check everything)
- EXPERTISE_FILE: .claude/commands/experts/crm/expertise.yaml
- MAX_LINES: 1000

## Instructions

- This is a self-improvement workflow for Kommo CRM expertise
- Focus on: contacts, leads, pipelines, tasks, notes, chat sync, OAuth
- Think of the expertise file as your mental model of the codebase
- Always validate against the real implementation
- Maintain YAML structure and enforce the line limit

## Workflow

### 1. Check Recent Changes (if CHECK_GIT_DIFF is true)
```bash
git diff --name-only HEAD~5 | grep -E 'kommo'
```

### 2. Read Current Expertise
Read EXPERTISE_FILE to understand current knowledge state

### 3. Validate Against Codebase
Key files to validate:
- services/kommo_crm/service.py
- services/kommo_crm/client.py
- services/kommo_crm/chats_client.py
- services/kommo_sync/
- core/interfaces/kommo_crm.py

### 4. Identify Discrepancies
Document what's:
- Outdated (API changes, new endpoints)
- Missing (new CRM operations, pipelines)
- Incorrect (wrong method signatures, stale descriptions)

### 5. Update Expertise File
Fix all identified issues while:
- Preserving YAML structure
- Keeping descriptions concise
- Adding new discoveries

### 6. Enforce Line Limit
```bash
wc -l .claude/commands/experts/crm/expertise.yaml
```
If over MAX_LINES, consolidate less-critical sections

### 7. Validate YAML Syntax
```bash
python3 -c "import yaml; yaml.safe_load(open('.claude/commands/experts/crm/expertise.yaml'))"
```

## Report Format

### Summary
- Files validated: N
- Discrepancies found: N
- Updates made: N
- Current line count: N / 1000

### Changes Made
- [List of specific changes]

### Recommendations
- [Any follow-up actions needed]
