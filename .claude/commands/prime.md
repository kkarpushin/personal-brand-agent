# Prime

Execute the `Run`, `Read` and `Report` sections to understand the codebase then summarize your understanding.

## Focus
- Primary focus: Understand the overall project structure and key patterns

## Run
git ls-files

## Read

### Core (always read these):
- @README.md
- @CLAUDE.md

### Template Infrastructure:
- @.claude/commands/plan.md
- @.claude/commands/build.md
- @.claude/commands/question.md
- @.claude/commands/experts/_template/expertise.yaml

### Salesbooster Project Files:

#### API Layer:
- @api/main.py
- @api/routes/webhooks.py
- @api/routes/messages.py
- @api/routes/contacts.py

#### Core Orchestration:
- @core/orchestrator/orchestrator.py
- @core/models/conversation.py
- @core/models/message.py
- @core/models/contact.py

#### Services:
- @services/linkedin_direct/service.py
- @services/conversation_analyzer/service.py
- @services/reply_generator/service.py
- @services/kommo_crm/service.py
- @services/calendly/service.py
- @services/telegram/service.py

#### Database:
- @database/models.py
- @database/repositories.py
- @database/connection.py

#### Configuration:
- @config/settings.py
- @env.example

## Report
Summarize your understanding of the codebase including:
- Project purpose and architecture
- Key directories and their roles
- Available commands in .claude/commands/
- Available agents in .claude/agents/
- Important patterns and conventions
- Any experts defined in .claude/commands/experts/
