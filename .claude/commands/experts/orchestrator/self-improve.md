---
allowed-tools: Read, Grep, Glob, Bash, Edit, Write, TodoWrite
description: Self-improve Orchestrator expertise by validating against codebase
argument-hint: [check_git_diff (true/false)] [focus_area (optional)]
---

# Orchestrator Expert - Self-Improve Mode

Maintain expertise accuracy by comparing against actual codebase implementation.

## Variables

- CHECK_GIT_DIFF: $1 (default: false)
- FOCUS_AREA: $2 (default: empty - check everything)
- EXPERTISE_FILE: .claude/commands/experts/orchestrator/expertise.yaml
- MAX_LINES: 1000

## Instructions

- This is a self-improvement workflow for Orchestrator expertise
- Focus on: event handlers, workflows, service coordination, core models
- Think of the expertise file as your mental model of the codebase
- Always validate against the real implementation
- Maintain YAML structure and enforce the line limit

## Workflow

### 1. Check Recent Changes (if CHECK_GIT_DIFF is true)
```bash
git diff --name-only HEAD~5 | grep -E 'orchestrator|core/models'
```

### 2. Read Current Expertise
Read EXPERTISE_FILE to understand current knowledge state

### 3. Validate Against Codebase
Key files to validate:
- core/orchestrator/orchestrator.py
- core/models/conversation.py
- core/models/message.py
- core/models/contact.py
- core/interfaces/

### 4. Identify Discrepancies
Document what's:
- Outdated (files moved, functions renamed)
- Missing (new event handlers, workflows)
- Incorrect (wrong line numbers, stale descriptions)

### 5. Update Expertise File
Fix all identified issues while:
- Preserving YAML structure
- Keeping descriptions concise
- Adding new discoveries

### 6. Enforce Line Limit
```bash
wc -l .claude/commands/experts/orchestrator/expertise.yaml
```
If over MAX_LINES, consolidate less-critical sections

### 7. Validate YAML Syntax
```bash
python3 -c "import yaml; yaml.safe_load(open('.claude/commands/experts/orchestrator/expertise.yaml'))"
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
