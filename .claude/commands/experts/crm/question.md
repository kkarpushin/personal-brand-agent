---
allowed-tools: Bash, Read, Grep, Glob, TodoWrite
description: Answer questions about Kommo CRM integration without coding
argument-hint: [question]
---

# CRM Expert - Question Mode

Answer the user's question by analyzing the Kommo CRM integration in Salesbooster. This is a question-answering task only - DO NOT write, edit, or create any files.

## Variables

- USER_QUESTION: $1
- EXPERTISE_PATH: .claude/commands/experts/crm/expertise.yaml

## Instructions

1. Focus on CRM functionality: contacts, leads, pipelines, tasks, notes, chat sync
2. Validate information from EXPERTISE_PATH against the actual codebase
3. If the question requires code changes, explain conceptually without implementing
4. Provide file paths and line numbers where relevant

## Workflow

1. Read the EXPERTISE_PATH file to get baseline understanding
2. Search codebase to validate and expand on expertise knowledge
3. Answer the question with supporting evidence

## Report Format

Provide your response as:

### Answer
[Direct answer to the USER_QUESTION]

### Evidence
- File: `path/to/file.py:line` - relevant code/pattern
- [Additional evidence points]

### Notes
[Any caveats, related considerations, or suggestions]
