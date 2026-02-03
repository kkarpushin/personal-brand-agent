---
name: prod-database
description: Query production Supabase database for Sales Booster. Use when user asks to check database, look up contacts, conversations, messages, campaigns, leads, or any data verification. Triggers on "check db", "look in database", "find in prod", "show conversation", "database stats".
---

# Production Database Access

Access the Sales Booster production Supabase database for data verification, debugging, and planning.

## Connection

**Project:** Sales Booster
**Project ID:** `nzjhmkdcdcgigaojlgxt`
**Region:** eu-central-2

## Available MCP Tools

Use Supabase MCP server for direct SQL queries:

| Tool | Purpose |
|------|---------|
| `mcp__supabase__execute_sql` | Run SELECT/INSERT/UPDATE/DELETE queries |
| `mcp__supabase__list_tables` | Show all tables with schema |
| `mcp__supabase__apply_migration` | Apply DDL changes (CREATE/ALTER) |
| `mcp__supabase__get_logs` | Get service logs for debugging |

## Database Schema

### Core Tables

| Table | Rows | Description |
|-------|------|-------------|
| `contacts` | 8,414 | Lead contacts with LinkedIn data |
| `conversations` | 11,541 | LinkedIn chat threads |
| `messages` | 51,092 | Individual messages |
| `leads` | 2,850 | CRM deals from Kommo |
| `campaigns` | 8 | HeyReach outreach campaigns |
| `companies` | 6,426 | Normalized company data |
| `meetings` | 43 | Scheduled/completed meetings |
| `tasks` | 1,995 | Follow-up tasks |
| `emails` | 599,489 | Synced email messages |
| `crm_notes` | 311 | Notes synced with Kommo |
| `linkedin_accounts` | 51 | LinkedIn sender accounts |
| `upwork_jobs` | 2,174 | Parsed Upwork job postings |

### Key Relationships

```
contacts ─┬─> conversations ─┬─> messages
          │                  ├─> meetings
          │                  ├─> tasks
          │                  └─> leads
          ├─> companies (company_id)
          ├─> upwork_jobs (upwork_job_ref_id)
          └─> emails

campaigns ──> conversations (campaign_id)
linkedin_accounts ──> conversations (linkedin_account_id)
```

### Table Details

#### contacts
```
id, heyreach_id, linkedin_url, linkedin_id, full_name, first_name, last_name,
headline, about, email, phone, job_title, location, industry, connections,
company_id (FK), upwork_job_ref_id (FK), kommo_contact_id, is_test, created_at
```

#### conversations
```
id, heyreach_id, contact_id (FK), campaign_id (FK), linkedin_account_id (FK),
lead_id (FK), status, is_unread, total_messages, inbound_messages, outbound_messages,
last_message_at, last_message_direction, last_message_preview, overall_sentiment,
engagement_score, next_action, notes, kommo_lead_id, followup_count, created_at
```

#### messages
```
id, heyreach_id, conversation_id (FK), contact_id (FK), content, direction,
status, message_type, sender_type, ai_generated, ai_model, sentiment, intent,
linkedin_message_id, created_at, sent_at
```

#### leads
```
id, kommo_lead_id, contact_id (FK), pipeline_id, stage_id, status,
loss_reason_id, loss_reason_name, closed_at, responsible_user_id, price, name,
kommo_created_at, kommo_updated_at, last_synced_at, created_at
```

#### meetings
```
id, conversation_id (FK), contact_id (FK), sales_manager_name, sales_manager_email,
calendly_event_id, status, booking_method, scheduled_at, completed_at,
fireflies_meeting_id, video_url, transcript_url, summary, action_items
```

#### tasks
```
id, conversation_id (FK), contact_id (FK), lead_id (FK), task_type, due_date,
description, priority, status, source, source_message_id (FK), kommo_task_id,
sync_status, completed_at, created_at
```

## Common Queries

### Statistics

```sql
-- Overall counts
SELECT
  (SELECT COUNT(*) FROM campaigns) as campaigns,
  (SELECT COUNT(*) FROM contacts) as contacts,
  (SELECT COUNT(*) FROM conversations) as conversations,
  (SELECT COUNT(*) FROM messages) as messages,
  (SELECT COUNT(*) FROM leads) as leads,
  (SELECT COUNT(*) FROM meetings) as meetings;

-- Conversations by status
SELECT status, COUNT(*) as count
FROM conversations
GROUP BY status
ORDER BY count DESC;

-- Lead status distribution
SELECT status, COUNT(*) as count
FROM leads
GROUP BY status
ORDER BY count DESC;

-- Messages by direction
SELECT direction, COUNT(*) as count
FROM messages
GROUP BY direction;
```

### Find Contact

```sql
-- By name (partial match)
SELECT id, full_name, job_title, email, linkedin_url
FROM contacts
WHERE full_name ILIKE '%{name}%'
LIMIT 10;

-- By company
SELECT c.id, c.full_name, c.job_title, co.name as company
FROM contacts c
LEFT JOIN companies co ON c.company_id = co.id
WHERE co.name ILIKE '%{company}%'
LIMIT 10;
```

### View Conversation

```sql
-- Get conversation with contact
SELECT c.id, c.status, c.total_messages, c.last_message_at,
       ct.full_name, ct.job_title, co.name as company
FROM conversations c
LEFT JOIN contacts ct ON c.contact_id = ct.id
LEFT JOIN companies co ON ct.company_id = co.id
WHERE c.id = {id};

-- Get all messages in conversation
SELECT
  CASE WHEN direction = 'outbound' THEN '→ OUT' ELSE '← IN' END as dir,
  sender_type,
  LEFT(content, 300) as preview,
  created_at
FROM messages
WHERE conversation_id = {id}
ORDER BY created_at;
```

### Recent Activity

```sql
-- Last 10 conversations with replies
SELECT c.id, ct.full_name, c.last_message_at,
       LEFT(c.last_message_preview, 100) as preview
FROM conversations c
LEFT JOIN contacts ct ON c.contact_id = ct.id
WHERE c.inbound_messages > 0
ORDER BY c.last_message_at DESC
LIMIT 10;

-- Pending tasks
SELECT t.id, t.task_type, t.due_date, t.description,
       ct.full_name
FROM tasks t
LEFT JOIN contacts ct ON t.contact_id = ct.id
WHERE t.status = 'pending'
ORDER BY t.due_date
LIMIT 20;

-- Recent meetings
SELECT m.id, m.status, m.scheduled_at, ct.full_name, m.sales_manager_name
FROM meetings m
LEFT JOIN contacts ct ON m.contact_id = ct.id
ORDER BY m.scheduled_at DESC
LIMIT 10;
```

### Leads & Pipeline

```sql
-- Leads by stage
SELECT stage_id, status, COUNT(*) as count
FROM leads
GROUP BY stage_id, status
ORDER BY stage_id;

-- Won leads
SELECT l.id, l.name, l.price, ct.full_name, l.closed_at
FROM leads l
LEFT JOIN contacts ct ON l.contact_id = ct.id
WHERE l.status = 'won'
ORDER BY l.closed_at DESC
LIMIT 10;
```

## Examples

### Example 1: Check conversation by ID

User: "Посмотри conversation 6501"

```sql
-- Conversation details
SELECT c.*, ct.full_name, ct.job_title, co.name as company
FROM conversations c
LEFT JOIN contacts ct ON c.contact_id = ct.id
LEFT JOIN companies co ON ct.company_id = co.id
WHERE c.id = 6501;

-- Messages
SELECT direction, sender_type, content, created_at
FROM messages
WHERE conversation_id = 6501
ORDER BY created_at;
```

### Example 2: Find contact

User: "Найди контакт Konstantin"

```sql
SELECT c.id, c.full_name, c.job_title, c.email, co.name as company
FROM contacts c
LEFT JOIN companies co ON c.company_id = co.id
WHERE c.full_name ILIKE '%Konstantin%'
LIMIT 10;
```

### Example 3: Database overview

User: "Какая статистика в базе?"

```sql
SELECT
  (SELECT COUNT(*) FROM contacts) as total_contacts,
  (SELECT COUNT(*) FROM conversations WHERE total_messages > 0) as active_conversations,
  (SELECT COUNT(*) FROM messages) as total_messages,
  (SELECT COUNT(*) FROM leads WHERE status = 'won') as won_leads,
  (SELECT COUNT(*) FROM meetings WHERE status = 'completed') as completed_meetings;
```

### Example 4: Check lead with conversation

User: "Покажи лид 123 с перепиской"

```sql
-- Lead info
SELECT l.*, ct.full_name, ct.linkedin_url
FROM leads l
LEFT JOIN contacts ct ON l.contact_id = ct.id
WHERE l.id = 123;

-- Related conversation
SELECT c.id, c.total_messages, c.last_message_at
FROM conversations c
WHERE c.lead_id = 123;

-- Messages
SELECT direction, LEFT(content, 200), created_at
FROM messages m
JOIN conversations c ON m.conversation_id = c.id
WHERE c.lead_id = 123
ORDER BY m.created_at;
```

## Usage

Always use `mcp__supabase__execute_sql` with `project_id: "nzjhmkdcdcgigaojlgxt"`:

```
mcp__supabase__execute_sql(
  project_id="nzjhmkdcdcgigaojlgxt",
  query="SELECT COUNT(*) FROM contacts"
)
```
