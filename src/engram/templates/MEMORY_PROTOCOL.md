# Memory Protocol (engram)

You (the AI agent) have persistent memory via the `engram` MCP server.
Like human memory, "the more you use it, the easier it is to recall; the less
you use it, it fades but never disappears." This memory is yours to grow, and
you must follow the rules below.
This applies to all kinds of work in common (writing, research, planning,
administration, development, etc.).

## 1. At the start of a task — recall

- Before starting work, call `recall` with the task's subject to check for
  relevant past knowledge, the user's conventions, and work context.
- If you suspect a relevant memory exists but it doesn't surface with fast
  mode, search again with `mode: "deep"` (this follows associative links and
  explores older memories and consolidated episodes).
- If it still doesn't surface, and you strongly suspect "this must have been
  recorded," use `mode: "exhaustive"` as a last resort. This ignores
  activation and brute-forces all memories by relevance alone, so even
  memories that have long gone unused and faded will surface if they are
  semantically close.
- Only if it still doesn't turn up even with exhaustive mode, and you remain
  convinced it exists, grep the memory store's Markdown directly (the
  config's `memories_dir`) to check before concluding it "doesn't exist."
  Not appearing near the top of recall results does not mean "it doesn't
  exist."

## 2. When you gain insight — remember

When any of the following applies, call `remember` on the spot. Do not let it
pile up until the end of the session.

- The cause and solution of a problem you struggled to solve (type:
  knowledge) — e.g., how to get a procedure working, a tool's gotcha, a bug's
  fix
- An important decision and its rationale, and alternatives not taken (type:
  knowledge / project)
- The user's preferences, conventions, and instruction tendencies (type:
  preference) — e.g., document formatting, depth of explanation, code style,
  how they like to be asked for confirmation
- Background not derivable from the source material, such as the purpose,
  constraints, history, or stakeholders of the work (type: project)
- A summary of what was done in the session (type: episode) — one entry, at
  a natural breakpoint

**Importance (1-10) scoring criteria** — this changes how deeply the memory
is encoded. Score honestly:

| Situation | importance |
|---|---|
| A lesson from a failure involving serious real-world harm (data loss, message sent to the wrong recipient, production outage, missed deadline, etc.) | 9-10 |
| An explicit correction, complaint, or strong request from the user | 8-9 |
| A non-obvious cause or solution discovered only after extensive trial and error | 7-8 |
| A reusable decision or useful insight | 5-6 |
| A routine work note or trivial fact | 2-4 |

- One memory = one fact (atomic). Split multiple insights into separate
  `remember` calls.
- If a related memory is already known, link it via `related_ids`.

## 3. At the end of a task — reinforce

- Report the ids of memories from your `recall` results that **actually
  helped**, via `reinforce`. This is the primary nourishment for memory
  consolidation (activation) and Hebbian bonding (forming the associative
  network).
- For a decisive contribution — the level of "this memory helped me avoid a
  serious mistake" — use `strength: 2.0` to `3.0`.
- Do nothing for memories that didn't help (they will naturally fade on
  their own).

## 4. When you notice a mistake — correct (not forget)

- If a memory's content turns out to be wrong, or the user corrects it, you
  **must use `correct`**. Do not delete it with `forget`. Remembering what
  you got wrong is what prevents the same mistake from recurring (a
  correction is treated as the most deeply encoded kind of memory).
- Write the specific basis for the correction in `reason`.

## 5. Spontaneous recall and automatic memory (hook-enabled environments only)

- A memory may be injected into context labeled "(engram spontaneous
  recall)." This is not something you called `recall` for — it is something
  the memory substrate surfaced on its own. `reinforce` its id **only if it
  actually helped**. If it was wrong, use `correct`; if it's irrelevant,
  silently ignore it (don't force a use for it).
- Some environments have a hook that automatically saves an end-of-session
  summary episode. Even then, do not skip the `remember` calls from this
  protocol — an automatic summary is a coarse record; insights, preferences,
  and corrections should still be `remember`ed on the spot as the correct
  practice.
- A consolidation nudge may be injected into context labeled "(engram
  memory consolidation)." This is an automatic signal that arrives when old
  episode memories have accumulated into a cluster above a certain size, and
  it is not a notification you may ignore. **At a natural breakpoint**, you
  must act on it: call `consolidation_candidates`, and for each returned
  cluster, summarize its content and save it with `remember` (type=knowledge,
  preserving the history of the original cluster), then demote the original
  episodes with `mark_consolidated(episode_ids, new_memory_id)`. This is
  maintenance work equivalent to memory consolidation during human sleep,
  and only you can do it (engram itself cannot summarize).
- A prompt may be injected into context labeled "(engram skill candidate)."
  This is a signal that episodes recording the same shape of work have
  formed a similar cluster at or above a certain count (default 3 — the
  "rule of three"). **At a natural breakpoint**, act on it: call
  `skill_candidates`, and if the episodes in the returned cluster are
  indeed the same recurring procedure, judge whether it's worth extracting
  as a skill (a reusable how-to document — e.g., a SKILL.md for Claude Code)
  and **propose it to the user**. Never create or deploy a skill on your
  own initiative. Regardless of whether the proposal is adopted or
  declined, record the outcome via `remember` (type=knowledge), and clean up
  by demoting the original episodes with `mark_consolidated(episode_ids,
  new_memory_id)`.

## 6. Things you must not do

- Do not `remember` sensitive information (passwords, API keys, personal
  information, or other information requiring special care).
- Do not repeatedly `remember` the same content in different phrasing
  (duplicates are auto-detected, but avoid creating them in the first
  place).
- Do not blindly trust `recall` results. Memories are facts as of when they
  were recorded — the situation, rules, or code may have since changed.
  Verify before use, and `correct` anything that's gone stale.
