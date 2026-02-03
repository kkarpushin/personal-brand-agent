---
allowed-tools: Read, Grep, Glob, Bash, Edit, Write, TodoWrite
description: Self-improve [DOMAIN] expertise by validating against codebase
argument-hint: [check_git_diff (true/false)] [focus_area (optional)]
---

# [DOMAIN] Expert - Self-Improve Mode

Maintain expertise accuracy by comparing against actual codebase implementation.

## Variables

- CHECK_GIT_DIFF: $1 (default: false)
- FOCUS_AREA: $2 (default: empty - check everything)
- EXPERTISE_FILE: .claude/commands/experts/_template/expertise.yaml
- MAX_LINES: 1000

## Instructions

- This is a self-improvement workflow
- Think of the expertise file as your mental model of the codebase
- Always validate against the real implementation
- Maintain YAML structure and enforce the line limit

## Workflow

### 1. Check Recent Changes (if CHECK_GIT_DIFF is true)
```bash
git diff --name-only HEAD~5 | grep -E '\.(py|ts|js|vue)$'
```

### 2. Read Current Expertise
Read EXPERTISE_FILE to understand current knowledge state

### 3. Validate Against Codebase
For each section in expertise.yaml:
- Verify file paths still exist
- Confirm line numbers are accurate
- Check function signatures match
- Validate patterns are still in use

### 4. Identify Discrepancies
Document what's:
- Outdated (files moved, functions renamed)
- Missing (new modules, patterns not documented)
- Incorrect (wrong line numbers, stale descriptions)

### 5. Update Expertise File
Fix all identified issues while:
- Preserving YAML structure
- Keeping descriptions concise
- Adding new discoveries

### 6. Enforce Line Limit
```bash
wc -l EXPERTISE_FILE
```
If over MAX_LINES, consolidate less-critical sections

### 7. Validate YAML Syntax
Ensure the file is valid YAML after edits

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
