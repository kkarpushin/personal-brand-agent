"""
Writer Agent for the LinkedIn Super Agent system.

Generates LinkedIn post drafts using type-specific templates, hooks, content
strategies, and CTA / hashtag selection logic.  Consumes an ``AnalysisBrief``
produced by the Analyzer and outputs a ``WriterOutput`` containing the primary
draft and any alternative drafts.

Architecture overview (from ``architecture.md`` lines 6533-6608):
    1. Select the best template for the content type
    2. Select hook style
    3. Build a generation prompt with extracted data and style guide
    4. Generate a full LinkedIn post draft via Claude
    5. Validate length and format constraints
    6. Generate alternative hooks
    7. Return ``WriterOutput`` with draft and metadata

Error philosophy: NO FALLBACKS.  If Claude fails to generate or the draft
violates constraints that cannot be self-corrected, raise ``WriterError``.

References:
    - architecture.md lines 6530-7704 (Writer Agent full spec)
    - architecture.md lines 7437-7567 (CTAs and hashtag strategies)
    - architecture.md lines 7572-7637 (Style guide)
"""

from __future__ import annotations

import json
import logging
import random
import re
from typing import Any, Dict, List, Optional

from src.exceptions import WriterError
from src.models import (
    AnalysisBrief,
    ContentType,
    DraftPost,
    HookStyle,
    CONTENT_TYPE_HOOK_STYLES,
    WriterOutput,
)
from src.tools.claude_client import ClaudeClient
from src.utils import generate_id, utc_now


logger = logging.getLogger("Writer")


# =============================================================================
# STYLE GUIDE
# =============================================================================

STYLE_GUIDE: Dict[str, Any] = {
    "phrases_to_use": [
        "Here's the thing:",
        "I've noticed that",
        "In my experience",
        "The real question is",
        "What surprised me most was",
    ],
    "phrases_to_avoid": [
        "In conclusion",
        "Furthermore",
        "It's worth noting that",
        "As an AI enthusiast",
        "Let me share my thoughts",
    ],
    "linkedin_rules": {
        "line_breaks": "After every 1-2 sentences",
        "max_emojis": 4,
        "hashtag_count": "3-5",
        "hook_max_chars": 210,
        "optimal_length": "1200-1500 chars",
        "max_length": 3000,
    },
}

# Recommended emojis per content type (architecture.md lines 7597-7607)
EMOJIS_BY_TYPE: Dict[ContentType, List[str]] = {
    ContentType.ENTERPRISE_CASE: ["\U0001f4ca", "\U0001f4a1", "\U0001f3af", "\U0001f4c8"],
    ContentType.PRIMARY_SOURCE: ["\U0001f52c", "\U0001f4d6", "\U0001f914", "\U0001f4ad"],
    ContentType.AUTOMATION_CASE: ["\u26a1", "\U0001f527", "\u23f1\ufe0f", "\U0001f4b0"],
    ContentType.COMMUNITY_CONTENT: ["\U0001f4a1", "\U0001f5e3\ufe0f", "\U0001f440", "\U0001f525"],
    ContentType.TOOL_RELEASE: ["\U0001f680", "\u2705", "\U0001f195", "\u26a1"],
}


# =============================================================================
# CONTENT TYPE -> TEMPLATE MAPPING
# =============================================================================

CONTENT_TYPE_TEMPLATE_MAPPING: Dict[str, Dict[str, Any]] = {
    ContentType.ENTERPRISE_CASE.value: {
        "primary_templates": ["metrics_story", "lessons_learned", "case_study"],
        "fallback_templates": ["insight_thread", "personal_story"],
    },
    ContentType.PRIMARY_SOURCE.value: {
        "primary_templates": ["contrarian", "explainer", "debate_starter"],
        "fallback_templates": ["insight_thread"],
    },
    ContentType.AUTOMATION_CASE.value: {
        "primary_templates": ["tutorial_light", "how_to", "results_story"],
        "fallback_templates": ["insight_thread", "list_post"],
    },
    ContentType.COMMUNITY_CONTENT.value: {
        "primary_templates": ["curated_insights", "list_post", "hot_take"],
        "fallback_templates": ["insight_thread", "question_based"],
    },
    ContentType.TOOL_RELEASE.value: {
        "primary_templates": ["first_look", "comparison", "implications"],
        "fallback_templates": ["insight_thread", "list_post"],
    },
}


# =============================================================================
# POST TEMPLATES
# =============================================================================

TEMPLATES: Dict[str, Dict[str, Any]] = {
    # ---- UNIVERSAL ----
    "insight_thread": {
        "name": "insight_thread",
        "structure": (
            "[HOOK - 1 line, attention grabbing]\n\n"
            "[CONTEXT - 2-3 lines, set the stage]\n\n"
            "[INSIGHT 1 - with brief explanation]\n\n"
            "[INSIGHT 2 - with brief explanation]\n\n"
            "[INSIGHT 3 - with brief explanation]\n\n"
            "[TAKEAWAY - 1-2 lines, so what?]\n\n"
            "[CTA - question or call to action]\n\n"
            "[HASHTAGS - 3-5 relevant]"
        ),
        "example_hooks": [
            "I spent 10 hours reading AI papers so you don't have to.",
            "The AI news you missed this week (but shouldn't have):",
            "Three things I learned about {topic} that changed how I think:",
        ],
        "tone": "educational, generous, expert",
        "best_for_types": ["primary_source", "community_content"],
        "length_target": "1200-1500 chars",
    },
    "list_post": {
        "name": "list_post",
        "structure": (
            "[HOOK - promise of value]\n\n"
            "[BRIEF CONTEXT - why this list matters]\n\n"
            "1. [ITEM 1 - with brief explanation]\n"
            "2. [ITEM 2 - with brief explanation]\n"
            "3. [ITEM 3 - with brief explanation]\n"
            "4. [ITEM 4 - with brief explanation]\n"
            "5. [ITEM 5 - with brief explanation]\n\n"
            "[BONUS - extra item or resource]\n\n"
            "[CTA - save this / which is your favorite?]\n\n"
            "[HASHTAGS]"
        ),
        "example_hooks": [
            "5 AI tools I use every day (and why):",
            "The top {N} lessons from {source}:",
            "{N} things nobody tells you about {topic}:",
        ],
        "tone": "practical, scannable, valuable",
        "best_for_types": ["community_content", "tool_release"],
        "length_target": "1200-1500 chars",
    },
    "personal_story": {
        "name": "personal_story",
        "structure": (
            "[STORY HOOK - something happened to me]\n\n"
            "[THE STORY - brief, specific details]\n\n"
            "[THE LESSON - what I learned]\n\n"
            "[THE INSIGHT - why this matters for you]\n\n"
            "[CTA - have you experienced this?]\n\n"
            "[HASHTAGS]"
        ),
        "example_hooks": [
            "Last week I made a mistake with AI that cost me {X}.",
            "I tried {thing} for 30 days. Here's what happened.",
            "A conversation with {person} completely changed my view on {topic}.",
        ],
        "tone": "vulnerable, authentic, relatable",
        "best_for_types": ["enterprise_case"],
        "length_target": "1000-1300 chars",
    },
    "contrarian": {
        "name": "contrarian",
        "structure": (
            "[PROVOCATIVE OPENING - challenge common belief]\n\n"
            "[ACKNOWLEDGE - why people think the opposite]\n\n"
            "[YOUR TAKE - with evidence]\n\n"
            "[NUANCE - it's not black and white]\n\n"
            "[INVITATION - what do you think?]\n\n"
            "[HASHTAGS]"
        ),
        "example_hooks": [
            "Unpopular opinion: {controversial_take}",
            "Everyone's excited about {topic}. Here's why I'm skeptical.",
            "Hot take: {bold_claim}",
        ],
        "tone": "confident, thoughtful, open to debate",
        "best_for_types": ["primary_source"],
        "length_target": "1000-1400 chars",
    },
    "question_based": {
        "name": "question_based",
        "structure": (
            "[PROVOCATIVE QUESTION - makes reader think]\n\n"
            "[WHY I'M ASKING - context]\n\n"
            "[PERSPECTIVE 1 - one way to look at it]\n\n"
            "[PERSPECTIVE 2 - another way]\n\n"
            "[MY CURRENT THINKING - where I lean]\n\n"
            "[INVITATION - genuinely curious what you think]\n\n"
            "[HASHTAGS]"
        ),
        "example_hooks": [
            "Is {assumption} actually true?",
            "What if we've been thinking about {topic} all wrong?",
            "Genuine question: {question}",
        ],
        "tone": "curious, humble, engaging",
        "best_for_types": ["community_content", "primary_source"],
        "length_target": "800-1200 chars",
    },
    "tutorial_light": {
        "name": "tutorial_light",
        "structure": (
            "[PROBLEM - relatable pain point]\n\n"
            "[SOLUTION PREVIEW - what you'll learn]\n\n"
            "[STEP 1]\n[STEP 2]\n[STEP 3]\n\n"
            "[RESULT - what this achieves]\n\n"
            "[BONUS TIP - extra value]\n\n"
            "[CTA - try it and let me know]\n\n"
            "[HASHTAGS]"
        ),
        "example_hooks": [
            "How to {achieve X} in 5 minutes (no code required):",
            "The {topic} cheat sheet I wish I had when starting:",
            "Stop doing {bad thing}. Do this instead:",
        ],
        "tone": "helpful, practical, encouraging",
        "best_for_types": ["automation_case"],
        "length_target": "1200-1600 chars",
    },
    # ---- ENTERPRISE CASE ----
    "metrics_story": {
        "name": "metrics_story",
        "structure": (
            "[METRIC HOOK - lead with impressive number]\n\n"
            "[COMPANY CONTEXT - who achieved this]\n\n"
            "[THE CHALLENGE - what problem they faced]\n\n"
            "[THE SOLUTION - what they implemented]\n\n"
            "[THE RESULTS - specific metrics]\n"
            "- Metric 1\n- Metric 2\n- Metric 3\n\n"
            "[THE LESSON - what we can learn]\n\n"
            "[CTA - have you seen similar results?]\n\n"
            "[HASHTAGS]"
        ),
        "example_hooks": [
            "{X}% improvement in {KPI}. Here's how {company} did it.",
            "From {before} to {after} in {timeline}. The story of {company}'s AI transformation.",
            "When {company} showed their board {metric}, everything changed.",
        ],
        "tone": "data-driven, credible, inspiring",
        "best_for_types": ["enterprise_case"],
        "length_target": "1300-1600 chars",
    },
    "lessons_learned": {
        "name": "lessons_learned",
        "structure": (
            "[EXPERIENCE HOOK - what they learned]\n\n"
            "[COMPANY INTRO - brief context]\n\n"
            "[LESSON 1 - what worked]\n\n"
            "[LESSON 2 - what didn't work]\n\n"
            "[LESSON 3 - what they'd do differently]\n\n"
            "[KEY TAKEAWAY - the meta-lesson]\n\n"
            "[CTA - what's your biggest AI implementation lesson?]\n\n"
            "[HASHTAGS]"
        ),
        "example_hooks": [
            "{company} spent {time} building {solution}. Here's what they'd do differently.",
            "The 3 biggest lessons from {company}'s AI journey.",
            "What {company} wishes they knew before implementing AI:",
        ],
        "tone": "reflective, honest, educational",
        "best_for_types": ["enterprise_case"],
        "length_target": "1200-1500 chars",
    },
    "case_study": {
        "name": "case_study",
        "structure": (
            "[OUTCOME HOOK - what they achieved]\n\n"
            "[COMPANY + PROBLEM]\n\n"
            "[SOLUTION OVERVIEW]\n\n"
            "[RESULTS]\n\n"
            "[WHY IT MATTERS - industry implications]\n\n"
            "[CTA - link to full case study or question]\n\n"
            "[HASHTAGS]"
        ),
        "example_hooks": [
            "Case study: How {company} achieved {result} with AI",
            "{industry} giant {company} just showed what's possible with AI.",
            "Inside {company}'s AI transformation: {result}",
        ],
        "tone": "professional, factual, impressive",
        "best_for_types": ["enterprise_case"],
        "length_target": "1400-1700 chars",
    },
    # ---- PRIMARY SOURCE ----
    "explainer": {
        "name": "explainer",
        "structure": (
            "[COMPLEXITY HOOK - I'll make this simple]\n\n"
            "[WHAT THE RESEARCH SAYS - core finding]\n\n"
            "[WHY IT MATTERS - so what?]\n\n"
            "[THE SIMPLE VERSION - analogy or plain language]\n\n"
            "[IMPLICATIONS - what this means for you]\n\n"
            "[MY TAKE - personal perspective]\n\n"
            "[CTA - does this change how you think about X?]\n\n"
            "[HASHTAGS]"
        ),
        "example_hooks": [
            "I read {paper} so you don't have to. The key insight:",
            "{complex_topic} explained in plain English:",
            "This research paper is 50 pages. Here's what actually matters:",
        ],
        "tone": "accessible, generous, clarifying",
        "best_for_types": ["primary_source"],
        "length_target": "1100-1400 chars",
    },
    "debate_starter": {
        "name": "debate_starter",
        "structure": (
            "[CONTROVERSIAL CLAIM - from the research]\n\n"
            "[THE EVIDENCE - what supports this]\n\n"
            "[THE COUNTERARGUMENT - what critics say]\n\n"
            "[THE NUANCE - it's complicated]\n\n"
            "[WHERE I LAND - my perspective]\n\n"
            "[INVITATION - what's your take?]\n\n"
            "[HASHTAGS]"
        ),
        "example_hooks": [
            "{author} claims {bold_claim}. Is this right?",
            "This paper will divide the AI community.",
            "Controversial take from {source}: {claim}",
        ],
        "tone": "intellectually engaging, balanced, provocative",
        "best_for_types": ["primary_source"],
        "length_target": "1000-1300 chars",
    },
    # ---- AUTOMATION CASE ----
    "how_to": {
        "name": "how_to",
        "structure": (
            "[OUTCOME HOOK - what you'll be able to do]\n\n"
            "[PREREQUISITES - what you need]\n\n"
            "[STEP 1] - {action}\n"
            "[STEP 2] - {action}\n"
            "[STEP 3] - {action}\n\n"
            "[RESULT] - what this achieves\n\n"
            "[PRO TIP] - insider knowledge\n\n"
            "[CTA - try it and tag me]\n\n"
            "[HASHTAGS]"
        ),
        "example_hooks": [
            "How I automated {task} with {tool}. Step by step:",
            "Build {solution} in {time}. No code required.",
            "The exact workflow I use to {achieve_result}:",
        ],
        "tone": "practical, detailed, empowering",
        "best_for_types": ["automation_case"],
        "length_target": "1300-1600 chars",
    },
    "results_story": {
        "name": "results_story",
        "structure": (
            "[RESULTS HOOK - the transformation]\n\n"
            "[BEFORE - the painful old way]\n\n"
            "[THE CHANGE - what I built/did]\n\n"
            "[AFTER - the new reality]\n\n"
            "[THE KEY INSIGHT - what made it work]\n\n"
            "[CTA - what would you automate first?]\n\n"
            "[HASHTAGS]"
        ),
        "example_hooks": [
            "From {hours} hours to {minutes} minutes. Here's how:",
            "This automation saves me {time} per {period}.",
            "{time_saved} saved. {cost_saved} saved. One workflow.",
        ],
        "tone": "results-oriented, specific, inspiring",
        "best_for_types": ["automation_case"],
        "length_target": "1100-1400 chars",
    },
    # ---- COMMUNITY CONTENT ----
    "curated_insights": {
        "name": "curated_insights",
        "structure": (
            "[CURATION HOOK - I found the gold]\n\n"
            "[SOURCE CONTEXT - where this came from]\n\n"
            "Insight 1: {insight}\n"
            "-> {attribution}\n\n"
            "Insight 2: {insight}\n"
            "-> {attribution}\n\n"
            "Insight 3: {insight}\n"
            "-> {attribution}\n\n"
            "[MY META-TAKEAWAY - connecting the dots]\n\n"
            "[CTA - what resonated with you?]\n\n"
            "[HASHTAGS]"
        ),
        "example_hooks": [
            "A {platform} thread about {topic} just exploded. Key takeaways:",
            "I spent {time} in {platform} threads. Here's the gold:",
            "The {platform} community is debating {topic}. Best insights:",
        ],
        "tone": "curator, synthesizer, community-connected",
        "best_for_types": ["community_content"],
        "length_target": "1200-1500 chars",
    },
    "hot_take": {
        "name": "hot_take",
        "structure": (
            "[HOT TAKE HOOK - the spicy opinion]\n\n"
            "[WHERE I SAW THIS - attribution]\n\n"
            "[THE ARGUMENT - why they might be right]\n\n"
            "[THE COUNTER - why they might be wrong]\n\n"
            "[MY POSITION - where I land]\n\n"
            "[INVITATION - convince me otherwise]\n\n"
            "[HASHTAGS]"
        ),
        "example_hooks": [
            "Someone on {platform} said: '{hot_take}'. They might be right.",
            "The most controversial {topic} take I've seen this week:",
            "Hot take from {source}: {claim}. Thoughts?",
        ],
        "tone": "engaged, opinionated but open, provocative",
        "best_for_types": ["community_content"],
        "length_target": "900-1200 chars",
    },
    # ---- TOOL RELEASE ----
    "first_look": {
        "name": "first_look",
        "structure": (
            "[BREAKING HOOK - something new dropped]\n\n"
            "[WHAT IT IS - tool overview]\n\n"
            "[KEY FEATURES]\n\n"
            "[FIRST IMPRESSION - my initial take]\n\n"
            "[WHO SHOULD CARE - target users]\n\n"
            "[ACCESS - how to try it]\n\n"
            "[CTA - are you going to try it?]\n\n"
            "[HASHTAGS]"
        ),
        "example_hooks": [
            "{company} just dropped {tool}. First impressions:",
            "New AI tool alert: {tool}. Here's what you need to know:",
            "I tested {tool} the moment it launched. Verdict:",
        ],
        "tone": "timely, informative, evaluative",
        "best_for_types": ["tool_release"],
        "length_target": "1200-1500 chars",
    },
    "comparison": {
        "name": "comparison",
        "structure": (
            "[COMPARISON HOOK - the matchup]\n\n"
            "[TOOL A OVERVIEW]\n\n"
            "[TOOL B OVERVIEW]\n\n"
            "[HEAD TO HEAD]\n\n"
            "[MY VERDICT - which one and why]\n\n"
            "[CTA - which do you prefer?]\n\n"
            "[HASHTAGS]"
        ),
        "example_hooks": [
            "{tool_a} vs {tool_b}: which one wins?",
            "I tested {tool_a} and {tool_b}. Clear winner.",
            "The {category} showdown: {tool_a} vs {tool_b}",
        ],
        "tone": "objective, analytical, helpful",
        "best_for_types": ["tool_release"],
        "length_target": "1300-1600 chars",
    },
    "implications": {
        "name": "implications",
        "structure": (
            "[SIGNIFICANCE HOOK - why this matters]\n\n"
            "[WHAT LAUNCHED - brief description]\n\n"
            "[WHO SHOULD CARE]\n\n"
            "[WHAT THIS MEANS FOR THE MARKET]\n\n"
            "[WHAT TO WATCH - future implications]\n\n"
            "[CTA - how does this affect your work?]\n\n"
            "[HASHTAGS]"
        ),
        "example_hooks": [
            "{tool} changes everything for {user_group}. Here's why:",
            "The {tool} announcement everyone's missing:",
            "What {company}'s new {tool} means for {industry}:",
        ],
        "tone": "strategic, forward-looking, analytical",
        "best_for_types": ["tool_release"],
        "length_target": "1100-1400 chars",
    },
}


# =============================================================================
# TYPE-SPECIFIC CTAs
# =============================================================================

TYPE_SPECIFIC_CTAS: Dict[ContentType, List[str]] = {
    ContentType.ENTERPRISE_CASE: [
        "Have you seen similar results in your organization?",
        "What's your biggest AI implementation lesson?",
        "Would this approach work in your industry?",
        "Comment with your company's AI journey.",
        "Tag someone leading AI transformation at your company.",
    ],
    ContentType.PRIMARY_SOURCE: [
        "Does this change how you think about this topic?",
        "What's your take on this?",
        "Agree or disagree? I'm curious.",
        "What research papers have shaped your thinking lately?",
        "Share your contrarian AI opinion below.",
    ],
    ContentType.AUTOMATION_CASE: [
        "Try it and let me know how it goes!",
        "What would you automate first?",
        "Tag someone who needs to see this workflow.",
        "Drop your automation wins in the comments.",
        "What's your favorite automation use case?",
    ],
    ContentType.COMMUNITY_CONTENT: [
        "What resonated most with you?",
        "Add your perspective to the discussion.",
        "What's your experience with this?",
        "The best insights often come from comments. Share yours.",
        "Which take do you agree with?",
    ],
    ContentType.TOOL_RELEASE: [
        "Are you going to try it?",
        "How does this compare to your current stack?",
        "Early adopters: share your first impressions.",
        "Which feature excites you most?",
        "Tag someone who should check this out.",
    ],
}


# =============================================================================
# TYPE-SPECIFIC HASHTAG STRATEGIES
# =============================================================================

TYPE_SPECIFIC_HASHTAGS: Dict[ContentType, Dict[str, List[str]]] = {
    ContentType.ENTERPRISE_CASE: {
        "broad": ["#AI", "#DigitalTransformation", "#Enterprise"],
        "specific": ["#AIImplementation", "#CaseStudy", "#AIinBusiness"],
    },
    ContentType.PRIMARY_SOURCE: {
        "broad": ["#AI", "#MachineLearning", "#Research"],
        "specific": ["#AIResearch", "#TechLeadership", "#FutureOfAI"],
    },
    ContentType.AUTOMATION_CASE: {
        "broad": ["#AI", "#Automation", "#Productivity"],
        "specific": ["#AITools", "#NoCode", "#WorkflowAutomation"],
    },
    ContentType.COMMUNITY_CONTENT: {
        "broad": ["#AI", "#TechCommunity"],
        "specific": ["#AITwitter", "#TechDiscussion", "#AICommunity"],
    },
    ContentType.TOOL_RELEASE: {
        "broad": ["#AI", "#TechNews", "#NewTools"],
        "specific": ["#AITools", "#ProductLaunch", "#TechReview"],
    },
}


# =============================================================================
# GENERATION PROMPTS BY CONTENT TYPE
# =============================================================================

BASE_GENERATION_CONSTRAINTS = (
    "CONSTRAINTS (apply to ALL content types):\n"
    "- Hook MUST fit in 210 chars (before 'see more' cutoff)\n"
    "- Maximum {max_emojis} emojis\n"
    "- Line breaks: After every 1-2 sentences for mobile readability\n"
    "- Hashtags: 3-5 total, mix of broad and specific\n"
    "- CTA: Must have clear call-to-action\n"
)

GENERATION_PROMPTS: Dict[ContentType, str] = {
    ContentType.ENTERPRISE_CASE: (
        "Write a LinkedIn post about this enterprise AI case study.\n\n"
        "CONTENT TYPE: Enterprise Case\n"
        "TEMPLATE: {template_name}\n"
        "TEMPLATE STRUCTURE:\n{template_structure}\n\n"
        "EXTRACTED DATA:\n"
        "- Company: {company}\n"
        "- Industry: {industry}\n"
        "- Challenge: {challenge}\n"
        "- Solution: {solution}\n"
        "- Key Metrics: {metrics}\n"
        "- Lessons Learned: {lessons}\n\n"
        "HOOKS (choose best or create variation):\n{hooks}\n\n"
        "INSIGHTS TO INCLUDE:\n{key_insights}\n\n"
        "STYLE REQUIREMENTS:\n"
        "- Lead with specific metrics when possible\n"
        "- Name the company (builds credibility)\n"
        "- Include concrete numbers, not vague claims\n"
        "- Make lessons actionable for readers\n"
        "- Use authoritative but approachable tone\n\n"
        "{constraints}\n\n"
        "- Length: {length_target}\n"
        "- End with: {suggested_cta}\n\n"
        "STYLE GUIDE:\n"
        "Phrases to use: {phrases_to_use}\n"
        "Phrases to avoid: {phrases_to_avoid}\n\n"
        "Generate the complete LinkedIn post now. Return ONLY the post text."
    ),
    ContentType.PRIMARY_SOURCE: (
        "Write a LinkedIn post explaining this research paper in EXTREMELY "
        "simple language — as if you are explaining it to a curious 10-year-old "
        "or a smart friend who knows nothing about AI.\n\n"
        "CONTENT TYPE: Primary Source (Research)\n"
        "TEMPLATE: {template_name}\n"
        "TEMPLATE STRUCTURE:\n{template_structure}\n\n"
        "EXTRACTED DATA:\n"
        "- Title/Source: {title}\n"
        "- Core Thesis: {thesis}\n"
        "- Key Finding: {key_finding}\n"
        "- Implications: {implications}\n\n"
        "HOOKS (choose best or create variation):\n{hooks}\n\n"
        "=== SIGNATURE STYLE: ELI5 RESEARCH BREAKDOWN ===\n"
        "This is the author's trademark: taking dense academic papers and "
        "making them dead simple. Follow these rules:\n\n"
        "1. EVERYDAY ANALOGY FIRST: Open the explanation with a vivid, "
        "everyday analogy that anyone can picture. Examples:\n"
        '   - "Imagine you are reading a book but your eyes keep jumping '
        'to the most interesting words. That is basically what attention does."\n'
        '   - "Think of it like a GPS that recalculates not just when you '
        'miss a turn, but learns which routes you personally prefer."\n'
        "   The analogy should make the reader think 'Oh, THAT is what "
        "this means.'\n\n"
        "2. ZERO JARGON IN THE MAIN BODY: Replace every technical term with "
        "a plain-language equivalent. If you must use a technical term, "
        "immediately explain it in parentheses.\n"
        "   - 'reinforcement learning' -> 'learning by trial and error'\n"
        "   - 'attention mechanism' -> 'the part that decides what to focus on'\n"
        "   - 'loss function' -> 'how the model measures its own mistakes'\n\n"
        "3. ONE CONCRETE EXAMPLE: Include at least one 'here is what this "
        "looks like in practice' scenario that grounds the research in "
        "something real and tangible.\n\n"
        "4. WHY SHOULD I CARE: Explicitly answer 'so what?' — why does "
        "this matter for a person who builds products, runs a business, "
        "or just uses technology daily?\n\n"
        "5. PAPER LINK AT THE END: Close with something like "
        "'The full paper is here: [source]. It gets more technical, but "
        "the core idea is exactly what I described above.' This positions "
        "the author as someone who reads the hard stuff so you don't have to.\n\n"
        "6. TONE: Curious, excited, generous with knowledge. Not academic. "
        "Not condescending. Like a smart friend who just read something "
        "cool and can't wait to tell you about it.\n\n"
        "{constraints}\n\n"
        "- Length: {length_target}\n"
        "- Simplification level: maximum — write for a general audience\n"
        "- End with: {suggested_cta}\n\n"
        "STYLE GUIDE:\n"
        "Phrases to use: {phrases_to_use}\n"
        "Phrases to avoid: {phrases_to_avoid}\n\n"
        "Generate the complete LinkedIn post now. Return ONLY the post text."
    ),
    ContentType.AUTOMATION_CASE: (
        "Write a LinkedIn post about this automation/workflow.\n\n"
        "CONTENT TYPE: Automation Case\n"
        "TEMPLATE: {template_name}\n"
        "TEMPLATE STRUCTURE:\n{template_structure}\n\n"
        "EXTRACTED DATA:\n"
        "- Task Automated: {task}\n"
        "- Tools Used: {tools}\n"
        "- Workflow Steps: {steps}\n"
        "- Time Saved: {time_saved}\n"
        "- Key Insight: {key_insight}\n\n"
        "HOOKS (choose best or create variation):\n{hooks}\n\n"
        "STYLE REQUIREMENTS:\n"
        "- Be specific about tools and steps\n"
        "- Include concrete time/cost savings\n"
        "- Make it feel replicable\n"
        "- Generous with details\n"
        "- Practical over philosophical\n\n"
        "{constraints}\n\n"
        "- Length: {length_target}\n"
        "- Must be actionable\n"
        "- End with: {suggested_cta}\n\n"
        "STYLE GUIDE:\n"
        "Phrases to use: {phrases_to_use}\n"
        "Phrases to avoid: {phrases_to_avoid}\n\n"
        "Generate the complete LinkedIn post now. Return ONLY the post text."
    ),
    ContentType.COMMUNITY_CONTENT: (
        "Write a LinkedIn post synthesizing this community discussion.\n\n"
        "CONTENT TYPE: Community Content\n"
        "TEMPLATE: {template_name}\n"
        "TEMPLATE STRUCTURE:\n{template_structure}\n\n"
        "EXTRACTED DATA:\n"
        "- Platform: {platform}\n"
        "- Topic: {topic}\n"
        "- Key Viewpoints: {viewpoints}\n"
        "- Notable Quotes: {quotes}\n"
        "- Practitioner Signals: {signals}\n\n"
        "HOOKS (choose best or create variation):\n{hooks}\n\n"
        "STYLE REQUIREMENTS:\n"
        "- Credit the community/sources\n"
        "- Capture diverse perspectives\n"
        "- Add your synthesis/take\n"
        "- Feel connected to the community\n"
        "- Invite continued discussion\n\n"
        "{constraints}\n\n"
        "- Length: {length_target}\n"
        "- Attribution is important\n"
        "- End with: {suggested_cta}\n\n"
        "STYLE GUIDE:\n"
        "Phrases to use: {phrases_to_use}\n"
        "Phrases to avoid: {phrases_to_avoid}\n\n"
        "Generate the complete LinkedIn post now. Return ONLY the post text."
    ),
    ContentType.TOOL_RELEASE: (
        "Write a LinkedIn post about this new AI tool/release.\n\n"
        "CONTENT TYPE: Tool Release\n"
        "TEMPLATE: {template_name}\n"
        "TEMPLATE STRUCTURE:\n{template_structure}\n\n"
        "EXTRACTED DATA:\n"
        "- Tool Name: {tool_name}\n"
        "- Company: {company}\n"
        "- Key Features: {features}\n"
        "- Target Users: {target_users}\n"
        "- Availability: {availability}\n"
        "- Comparison: {comparison}\n\n"
        "HOOKS (choose best or create variation):\n{hooks}\n\n"
        "STYLE REQUIREMENTS:\n"
        "- Feel timely/fresh\n"
        "- Be specific about features\n"
        "- Help readers evaluate fit\n"
        "- Balanced (pros and cons)\n"
        "- Include access/try info\n\n"
        "{constraints}\n\n"
        "- Length: {length_target}\n"
        "- Include demo/link if available\n"
        "- End with: {suggested_cta}\n\n"
        "STYLE GUIDE:\n"
        "Phrases to use: {phrases_to_use}\n"
        "Phrases to avoid: {phrases_to_avoid}\n\n"
        "Generate the complete LinkedIn post now. Return ONLY the post text."
    ),
}


# =============================================================================
# WRITER AGENT
# =============================================================================


class WriterAgent:
    """Generate LinkedIn post drafts using type-specific templates and Claude.

    The Writer Agent consumes an ``AnalysisBrief`` and produces a
    ``WriterOutput`` containing a primary draft, alternative hooks, and
    metadata about template / hook selection decisions.

    Args:
        claude: Async Claude API client used for generation.
    """

    def __init__(self, claude: ClaudeClient) -> None:
        self.claude = claude
        self.logger = logging.getLogger("Writer")

    # -----------------------------------------------------------------
    # PUBLIC INTERFACE
    # -----------------------------------------------------------------

    async def run(
        self,
        analysis_brief: AnalysisBrief,
        content_type: ContentType,
        preferred_templates: Optional[List[str]] = None,
        hook_styles: Optional[List[HookStyle]] = None,
        cta_style: Optional[str] = None,
        revision_instructions: Optional[List[str]] = None,
    ) -> WriterOutput:
        """Generate a LinkedIn post draft from an analysis brief.

        Args:
            analysis_brief: Structured brief produced by the Analyzer.
            content_type: The classified content type for this topic.
            preferred_templates: Caller-specified template preferences.  If
                ``None`` the mapping for *content_type* is used.
            hook_styles: Desired hook styles.  Defaults to those mapped to
                *content_type* via ``CONTENT_TYPE_HOOK_STYLES``.
            cta_style: Override CTA text.  If ``None`` a random type-specific
                CTA is selected.
            revision_instructions: Optional list of revision notes from QC to
                incorporate into the generation prompt.

        Returns:
            A ``WriterOutput`` dataclass with the primary draft, alternative
            drafts list, template category, and generation metadata.

        Raises:
            WriterError: If generation or validation fails.
        """

        if preferred_templates is None:
            preferred_templates = self._get_templates_for_type(content_type)
        if hook_styles is None:
            hook_styles = CONTENT_TYPE_HOOK_STYLES.get(content_type, [])

        # 1. Select template
        template = self._select_template(content_type, preferred_templates)
        self.logger.info(
            "Selected template '%s' for content type '%s'",
            template["name"],
            content_type.value,
        )

        # 2. Select hook style
        hook_style = self._select_hook_style(content_type, hook_styles)
        self.logger.info("Selected hook style '%s'", hook_style.value)

        # 3. Select CTA
        selected_cta = cta_style or self._select_cta(content_type)

        # 4. Build generation prompt
        prompt = self._build_prompt(
            analysis_brief,
            content_type,
            template,
            hook_style,
            selected_cta,
            revision_instructions,
        )

        # 5. Generate with Claude -- NO FALLBACK, fail fast
        try:
            response = await self.claude.generate(
                prompt,
                system=(
                    "You are an expert LinkedIn content writer. "
                    "Your signature skill is explaining complex research "
                    "in absurdly simple language — everyday analogies, "
                    "zero jargon, vivid examples. You make people feel "
                    "smart, not stupid."
                ),
                max_tokens=4096,
                temperature=0.7,
            )
        except Exception as exc:
            raise WriterError(
                f"Claude generation failed for content type "
                f"'{content_type.value}': {exc}"
            ) from exc

        # 6. Parse response into DraftPost
        draft = self._parse_draft(
            response,
            analysis_brief,
            content_type,
            template,
            hook_style,
        )

        # 7. Validate constraints
        self._validate_draft(draft)

        # 8. Generate alternative hooks
        alt_hooks = await self._generate_alt_hooks(
            analysis_brief, content_type, draft.hook
        )

        # 9. Build alternative drafts from alt hooks
        alternatives: List[DraftPost] = []
        for alt_hook in alt_hooks:
            alt_draft = DraftPost(
                hook=alt_hook,
                body=draft.body,
                cta=draft.cta,
                hashtags=draft.hashtags,
                full_text=alt_hook + "\n\n" + draft.body + "\n\n" + draft.cta,
                template_used=draft.template_used,
                hook_style=draft.hook_style,
                content_type=draft.content_type,
                character_count=len(alt_hook) + len(draft.body) + len(draft.cta) + 4,
                visual_brief=draft.visual_brief,
                visual_type=draft.visual_type,
                key_terms=draft.key_terms,
            )
            alternatives.append(alt_draft)

        # 10. Assemble WriterOutput (matches models.py WriterOutput)
        return WriterOutput(
            primary_draft=draft,
            alternatives=alternatives,
            template_category=self._template_category(template, content_type),
            generation_metadata={
                "template_used": template["name"],
                "hook_style": hook_style.value,
                "cta_selected": selected_cta,
                "alternative_hooks": alt_hooks,
                "alternative_templates": [
                    t for t in preferred_templates if t != template["name"]
                ],
                "template_selection_rationale": (
                    f"Selected '{template['name']}' for {content_type.value}"
                ),
                "hook_selection_rationale": (
                    f"Used {hook_style.value} style hook"
                ),
                "confidence_score": 0.8,
                "areas_of_uncertainty": [],
                "revision_instructions_applied": revision_instructions or [],
            },
        )

    # -----------------------------------------------------------------
    # TEMPLATE SELECTION
    # -----------------------------------------------------------------

    @staticmethod
    def _get_templates_for_type(content_type: ContentType) -> List[str]:
        """Return primary + fallback template names for *content_type*."""

        mapping = CONTENT_TYPE_TEMPLATE_MAPPING.get(content_type.value, {})
        primaries: List[str] = mapping.get("primary_templates", [])
        fallbacks: List[str] = mapping.get("fallback_templates", [])
        return primaries + fallbacks

    def _select_template(
        self,
        content_type: ContentType,
        preferred_templates: List[str],
    ) -> Dict[str, Any]:
        """Choose the best template, preferring caller choices then primaries.

        If a preferred template exists in ``TEMPLATES`` and fits
        *content_type*, it is returned.  Otherwise, the first primary
        template for the type is returned.

        Raises:
            WriterError: If no valid template can be resolved.
        """

        # Try preferred templates in order
        for name in preferred_templates:
            if name in TEMPLATES:
                return TEMPLATES[name]

        # Fallback: first primary template for the type
        mapping = CONTENT_TYPE_TEMPLATE_MAPPING.get(content_type.value, {})
        for name in mapping.get("primary_templates", []):
            if name in TEMPLATES:
                return TEMPLATES[name]

        raise WriterError(
            f"No valid template found for content type '{content_type.value}'. "
            f"Tried: {preferred_templates}"
        )

    # -----------------------------------------------------------------
    # HOOK STYLE SELECTION
    # -----------------------------------------------------------------

    @staticmethod
    def _select_hook_style(
        content_type: ContentType,
        hook_styles: List[HookStyle],
    ) -> HookStyle:
        """Pick a hook style from the provided list.

        Prefers styles that are mapped to *content_type*, then falls back to
        a random choice from the provided list.

        Raises:
            WriterError: If *hook_styles* is empty and no defaults exist.
        """

        allowed = CONTENT_TYPE_HOOK_STYLES.get(content_type, [])
        matching = [hs for hs in hook_styles if hs in allowed]

        if matching:
            return random.choice(matching)
        if hook_styles:
            return random.choice(hook_styles)
        if allowed:
            return random.choice(allowed)
        raise WriterError(
            f"No hook styles available for content type '{content_type.value}'"
        )

    # -----------------------------------------------------------------
    # CTA SELECTION
    # -----------------------------------------------------------------

    @staticmethod
    def _select_cta(content_type: ContentType) -> str:
        """Return a random CTA string appropriate for *content_type*."""

        ctas = TYPE_SPECIFIC_CTAS.get(content_type, [])
        if ctas:
            return random.choice(ctas)
        return "What are your thoughts?"

    # -----------------------------------------------------------------
    # HASHTAG SELECTION
    # -----------------------------------------------------------------

    @staticmethod
    def _select_hashtags(content_type: ContentType) -> List[str]:
        """Build a hashtag list (3-5) mixing broad and specific tags."""

        strategy = TYPE_SPECIFIC_HASHTAGS.get(content_type, {})
        broad: List[str] = strategy.get("broad", ["#AI"])
        specific: List[str] = strategy.get("specific", [])

        # Pick 1-2 broad + 2-3 specific to hit the 3-5 target
        selected_broad = random.sample(broad, min(2, len(broad)))
        selected_specific = random.sample(specific, min(3, len(specific)))

        hashtags = selected_broad + selected_specific
        # Ensure uniqueness while preserving order
        seen: set[str] = set()
        unique: List[str] = []
        for tag in hashtags:
            if tag not in seen:
                seen.add(tag)
                unique.append(tag)
        return unique[:5]

    # -----------------------------------------------------------------
    # PROMPT BUILDING
    # -----------------------------------------------------------------

    def _build_prompt(
        self,
        brief: AnalysisBrief,
        content_type: ContentType,
        template: Dict[str, Any],
        hook_style: HookStyle,
        cta: str,
        revision_instructions: Optional[List[str]],
    ) -> str:
        """Assemble the full generation prompt for Claude."""

        # Resolve constraints string
        constraints = BASE_GENERATION_CONSTRAINTS.format(
            max_emojis=STYLE_GUIDE["linkedin_rules"]["max_emojis"],
        )

        # Build hook examples string
        hooks_str = "\n".join(
            f"  - {h}" for h in template.get("example_hooks", [])
        )

        # Common substitution values
        common: Dict[str, str] = {
            "template_name": template["name"],
            "template_structure": template["structure"],
            "hooks": hooks_str,
            "key_insights": "\n".join(
                f"  - {f}" for f in brief.key_findings
            ),
            "length_target": template.get("length_target", "1200-1500 chars"),
            "suggested_cta": cta,
            "constraints": constraints,
            "phrases_to_use": ", ".join(STYLE_GUIDE["phrases_to_use"]),
            "phrases_to_avoid": ", ".join(STYLE_GUIDE["phrases_to_avoid"]),
            "complexity_level": brief.complexity_level,
        }

        # Type-specific data extraction
        ext = brief.extraction_data.extracted_fields if brief.extraction_data else {}

        type_values = self._extract_type_values(content_type, brief, ext)
        common.update(type_values)

        # Pick the right prompt template
        prompt_template = GENERATION_PROMPTS.get(content_type)
        if prompt_template is None:
            raise WriterError(
                f"No generation prompt defined for content type "
                f"'{content_type.value}'"
            )

        # Safe format: ignore missing keys by using a default-returning dict
        prompt = _safe_format(prompt_template, common)

        # Append revision instructions if present
        if revision_instructions:
            revision_block = (
                "\n\nREVISION INSTRUCTIONS (incorporate these changes):\n"
                + "\n".join(f"  - {r}" for r in revision_instructions)
            )
            prompt += revision_block

        return prompt

    @staticmethod
    def _extract_type_values(
        content_type: ContentType,
        brief: AnalysisBrief,
        ext: Dict[str, Any],
    ) -> Dict[str, str]:
        """Pull type-specific values from the extraction data and brief."""

        values: Dict[str, str] = {}

        if content_type == ContentType.ENTERPRISE_CASE:
            values["company"] = ext.get("company", "Unknown Company")
            values["industry"] = ext.get("industry", "Technology")
            values["challenge"] = ext.get("problem_domain", brief.main_argument)
            values["solution"] = ext.get("solution", "AI implementation")
            values["metrics"] = json.dumps(ext.get("metrics", {}))
            values["lessons"] = ", ".join(ext.get("lessons_learned", []))

        elif content_type == ContentType.PRIMARY_SOURCE:
            values["title"] = brief.title
            values["thesis"] = brief.main_argument
            values["key_finding"] = (
                brief.key_findings[0] if brief.key_findings else "N/A"
            )
            values["implications"] = ext.get(
                "counterintuitive_finding", "Significant implications"
            )

        elif content_type == ContentType.AUTOMATION_CASE:
            values["task"] = ext.get("use_case_domain", "workflow task")
            values["tools"] = ", ".join(ext.get("integrations", []))
            values["steps"] = ", ".join(ext.get("workflow_components", []))
            values["time_saved"] = ext.get("time_saved", "significant time")
            values["key_insight"] = (
                brief.key_findings[0] if brief.key_findings else "N/A"
            )

        elif content_type == ContentType.COMMUNITY_CONTENT:
            values["platform"] = ext.get("platform", "the community")
            values["topic"] = brief.title
            values["viewpoints"] = ", ".join(
                ext.get("key_contributors", [])
            )
            values["quotes"] = ext.get("notable_quotes", "")
            values["signals"] = str(ext.get("engagement_metrics", {}))

        elif content_type == ContentType.TOOL_RELEASE:
            values["tool_name"] = ext.get("tool_name", "New Tool")
            values["company"] = ext.get("company", "Unknown")
            values["features"] = ", ".join(ext.get("key_features", []))
            values["target_users"] = ext.get("target_users", "developers")
            values["availability"] = ext.get("pricing_model", "See website")
            values["comparison"] = ", ".join(ext.get("competing_tools", []))

        return values

    # -----------------------------------------------------------------
    # RESPONSE PARSING
    # -----------------------------------------------------------------

    def _parse_draft(
        self,
        raw_text: str,
        brief: AnalysisBrief,
        content_type: ContentType,
        template: Dict[str, Any],
        hook_style: HookStyle,
    ) -> DraftPost:
        """Parse raw Claude output into a ``DraftPost`` dataclass."""

        text = raw_text.strip()

        # Split into lines for hook extraction
        lines = [ln for ln in text.split("\n") if ln.strip()]
        hook = lines[0] if lines else ""

        # Extract hashtags from the end
        hashtags = self._extract_hashtags(text)
        if not hashtags:
            hashtags = self._select_hashtags(content_type)

        # Remove hashtags from body text for clean separation
        body_text = text
        for tag in hashtags:
            body_text = body_text.replace(tag, "").strip()

        # Attempt to separate CTA (last non-hashtag paragraph)
        paragraphs = [p.strip() for p in body_text.split("\n\n") if p.strip()]
        if len(paragraphs) >= 3:
            cta = paragraphs[-1]
            body = "\n\n".join(paragraphs[1:-1])
        elif len(paragraphs) == 2:
            cta = paragraphs[-1]
            body = paragraphs[0]
        else:
            cta = self._select_cta(content_type)
            body = body_text

        # Build full_text with hashtags appended
        full_text = text if hashtags else f"{text}\n\n{' '.join(hashtags)}"

        # Build visual brief from content type
        visual_type = brief.recommended_visual_type or "quote_card"
        visual_brief = (
            f"Create a {visual_type} visual for a {content_type.value} "
            f"post about: {brief.title}"
        )

        return DraftPost(
            hook=hook,
            body=body,
            cta=cta,
            hashtags=hashtags,
            full_text=full_text,
            template_used=template["name"],
            hook_style=hook_style,
            content_type=content_type,
            character_count=len(full_text),
            visual_brief=visual_brief,
            visual_type=visual_type,
            key_terms=brief.key_findings[:5],
        )

    @staticmethod
    def _extract_hashtags(text: str) -> List[str]:
        """Extract hashtags from the generated text."""

        return re.findall(r"#\w+", text)

    # -----------------------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------------------

    def _validate_draft(self, draft: DraftPost) -> None:
        """Validate that the draft meets LinkedIn constraints.

        Raises:
            WriterError: If any hard constraint is violated.
        """

        issues: List[str] = []

        # Hook length check
        hook_limit = STYLE_GUIDE["linkedin_rules"]["hook_max_chars"]
        if len(draft.hook) > hook_limit:
            issues.append(
                f"Hook exceeds {hook_limit} chars "
                f"(actual: {len(draft.hook)} chars)"
            )

        # Total length check
        max_length = STYLE_GUIDE["linkedin_rules"]["max_length"]
        if draft.character_count > max_length:
            issues.append(
                f"Post exceeds {max_length} chars "
                f"(actual: {draft.character_count} chars)"
            )

        if draft.character_count < 200:
            issues.append(
                f"Post is too short ({draft.character_count} chars). "
                f"Minimum useful length is ~200 chars."
            )

        # Hashtag count check
        if len(draft.hashtags) < 3:
            self.logger.warning(
                "Post has fewer than 3 hashtags (%d). Recommended: 3-5.",
                len(draft.hashtags),
            )
        if len(draft.hashtags) > 5:
            self.logger.warning(
                "Post has more than 5 hashtags (%d). Recommended: 3-5.",
                len(draft.hashtags),
            )

        # Emoji count check
        emoji_pattern = re.compile(
            "[\U0001f300-\U0001f9ff\u2600-\u26ff\u2700-\u27bf"
            "\ufe00-\ufe0f\u200d]+",
            flags=re.UNICODE,
        )
        emoji_count = len(emoji_pattern.findall(draft.full_text))
        max_emojis = STYLE_GUIDE["linkedin_rules"]["max_emojis"]
        if emoji_count > max_emojis:
            self.logger.warning(
                "Post contains %d emojis (max recommended: %d).",
                emoji_count,
                max_emojis,
            )

        # Check for phrases to avoid
        lower_text = draft.full_text.lower()
        for phrase in STYLE_GUIDE["phrases_to_avoid"]:
            if phrase.lower() in lower_text:
                self.logger.warning(
                    "Post contains discouraged phrase: '%s'", phrase
                )

        # Hard failures
        if issues:
            raise WriterError(
                f"Draft validation failed: {'; '.join(issues)}"
            )

        self.logger.info(
            "Draft validated: %d chars, hook %d chars, %d hashtags",
            draft.character_count,
            len(draft.hook),
            len(draft.hashtags),
        )

    # -----------------------------------------------------------------
    # ALTERNATIVE HOOK GENERATION
    # -----------------------------------------------------------------

    async def _generate_alt_hooks(
        self,
        brief: AnalysisBrief,
        content_type: ContentType,
        primary_hook: str,
    ) -> List[str]:
        """Generate 2-3 alternative hook lines via Claude.

        Returns:
            List of alternative hook strings.  Returns an empty list (with
            a warning log) if generation fails -- this is a non-critical
            enhancement, but we still log the failure clearly.
        """

        prompt = (
            f"Generate 3 alternative LinkedIn post hooks for a "
            f"{content_type.value} post about: {brief.title}\n\n"
            f"The primary hook is: \"{primary_hook}\"\n\n"
            f"Each alternative MUST:\n"
            f"- Be under 210 characters\n"
            f"- Use a different angle than the primary\n"
            f"- Be compelling and stop the scroll\n\n"
            f"Return ONLY the 3 hooks, one per line, no numbering or bullets."
        )

        try:
            response = await self.claude.generate(
                prompt,
                system="You are an expert LinkedIn copywriter.",
                max_tokens=512,
                temperature=0.9,
            )
            hooks = [
                line.strip()
                for line in response.strip().split("\n")
                if line.strip() and len(line.strip()) <= 210
            ]
            return hooks[:3]
        except Exception as exc:
            self.logger.warning(
                "Alternative hook generation failed (non-critical): %s", exc
            )
            return []

    # -----------------------------------------------------------------
    # HELPERS
    # -----------------------------------------------------------------

    @staticmethod
    def _template_category(
        template: Dict[str, Any], content_type: ContentType
    ) -> str:
        """Derive a category label for the template."""

        best_for = template.get("best_for_types", [])
        if content_type.value in best_for:
            return content_type.value
        return "universal"


# =============================================================================
# MODULE-PRIVATE HELPERS
# =============================================================================


class _DefaultDict(dict):  # type: ignore[type-arg]
    """A dict subclass that returns ``'{key}'`` for missing keys.

    Used by ``_safe_format`` to avoid ``KeyError`` when the prompt template
    references a placeholder that was not supplied.
    """

    def __missing__(self, key: str) -> str:
        return f"{{{key}}}"


def _safe_format(template: str, values: Dict[str, str]) -> str:
    """Format *template* with *values*, leaving missing placeholders intact."""

    return template.format_map(_DefaultDict(values))


# =============================================================================
# FACTORY
# =============================================================================


async def create_writer() -> WriterAgent:
    """Factory function to create a ``WriterAgent`` with default clients."""
    claude = ClaudeClient()
    return WriterAgent(claude=claude)
