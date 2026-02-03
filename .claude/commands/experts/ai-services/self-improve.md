---
allowed-tools: Read, Grep, Glob, Bash, Edit, Write, TodoWrite
description: Self-improve AI Services expertise by validating against codebase
argument-hint: [check_git_diff (true/false)] [focus_area (optional)]
---

# AI Services Expert - Self-Improve Mode

Maintain expertise accuracy by comparing against actual codebase implementation.

## Variables

- CHECK_GIT_DIFF: $1 (default: false)
- FOCUS_AREA: $2 (default: empty - check everything)
- EXPERTISE_FILE: .claude/commands/experts/ai-services/expertise.yaml
- MAX_LINES: 1000

## Instructions

- This is a self-improvement workflow for AI Services expertise
- Focus on: conversation analysis, reply generation, message analysis, knowledge services
- Think of the expertise file as your mental model of the codebase
- Always validate against the real implementation
- Maintain YAML structure and enforce the line limit

## Workflow

### 1. Check Recent Changes (if CHECK_GIT_DIFF is true)
```bash
git diff --name-only HEAD~5 | grep -E 'conversation_analyzer|reply_generator|message_analyzer|ai_messages|knowledge'
```

### 2. Read Current Expertise
Read EXPERTISE_FILE to understand current knowledge state

### 3. Validate Against Codebase
Key files to validate:
- services/conversation_analyzer/service.py
- services/reply_generator/service.py
- services/message_analyzer/
- services/ai_messages/
- services/knowledge/

### 4. Identify Discrepancies
Document what's:
- Outdated (prompts changed, models updated)
- Missing (new analysis features, data classes)
- Incorrect (wrong method signatures, stale descriptions)

### 5. Update Expertise File
Fix all identified issues while:
- Preserving YAML structure
- Keeping descriptions concise
- Adding new discoveries

### 6. Enforce Line Limit
```bash
wc -l .claude/commands/experts/ai-services/expertise.yaml
```
If over MAX_LINES, consolidate less-critical sections

### 7. Validate YAML Syntax
```bash
python3 -c "import yaml; yaml.safe_load(open('.claude/commands/experts/ai-services/expertise.yaml'))"
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
