# engram Initial Interview (Onboarding)

An interview script for registering the user's basic profile, mindset, and
working conventions into engram at the start. Usable regardless of
occupation (developer, researcher, administrative staff, planner, manager,
etc.).

## How to use

Ask any agent (Claude Code / Codex / Antigravity) the following:

> Read KnowledgeBase\ONBOARDING.md and interview me.
> Save each answer to engram with `remember`, one fact per memory.

You don't have to do it all at once. It can be done in multiple sessions,
one category at a time (engram automatically detects and merges
duplicates).

## Instructions for the agent

- First ask Category 1 (Profile), then **tailor the rest of the questions to
  the person's line of work based on their answers** (ask developers about
  coding conventions, ask document-centric people about formatting, ask
  people who do both about both).
- Break answers down into **one fact = one memory** and call `remember` on
  each.
  - type: the person's own attributes/preferences/conventions →
    `preference` / context of ongoing work → `project`
  - importance: strong conventions that shape how work should be done → 8,
    general preferences → 6, minor reference-level info → 4
  - Tag with the category name (profile / style / communication / projects /
    ng)
- For vague answers, ask for a concrete example before saving
  ("I like clear documents" → what makes a document clear to you?)
- At the end, present the list of saved memories, and fix any errors with
  `correct`.

## Categories and questions

### 1. Profile (tags: profile)
- What's your role/occupation/organizational context? (e.g., researcher,
  engineer, administrative staff, manager)
- What does your work mainly consist of? (writing, research, analysis,
  development, coordination, etc. — rough proportions are useful too)
- What's your proficiency level in each area? (used to judge how deep
  explanations should go — e.g., "technical terms are fine" vs. "outside my
  field, keep it simple")
- What's your usual work environment? (OS, apps/tools/services you commonly
  use)

### 2. Work conventions (tags: style) — tailor to the person's line of work
- What are the formatting/style rules for deliverables?
  (Documents: format, addressing, dates, numbering conventions, and other
  organizational rules / Development: language, code style, testing
  philosophy / Materials: preferred structure, tone, length)
- What's your preferred way of working? (plan first then act, vs. act and
  adjust as you go; want to see drafts early, vs. want to see things after
  they're more polished)
- What routine tasks do you repeat often? (and any procedures/templates for
  them)

### 3. Communication preferences (tags: communication)
- Do you prefer conclusions first, or the reasoning/background first?
- Do you want proactive suggestions, or only exactly what was instructed?
- Where's the line between "check with me first" and "just go ahead"?
- What are your preferences for language, tone, and how you'd like to be
  addressed?

### 4. Ongoing work (tags: projects, type: project)
- What projects are currently active, and what are their goals/deadlines?
- What should be known about the people/organizations involved? (who the
  deliverable is for, the approval process, people to be mindful of)

### 5. Things you don't want (tags: ng, importance: 8 or higher)
- Is there anything you absolutely don't want done? (especially if there's
  been a past incident)
- Is there any information or topic that needs to be handled with
  particular care?
