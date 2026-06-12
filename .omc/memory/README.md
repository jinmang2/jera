# Project memory snapshot (for session continuity)

These files mirror this machine's Claude Code auto-memory so the project's accumulated
context travels with the repo and can be restored on a new machine/session.

- `MEMORY.md` — the memory index (one line per memory).
- `jera-rag-architecture.md` — the durable project memory: architecture decisions, shipped
  milestones (M1–M5a), env constraints, and the next-milestone (M5b) plan.

## Restore on a new machine

Copy these into the Claude Code project-memory directory for this repo path:

```bash
DEST="$HOME/.claude/projects/$(pwd | sed 's#/#-#g')/memory"
mkdir -p "$DEST" && cp .omc/memory/*.md "$DEST/"
```

(Claude Code loads `MEMORY.md` into context each session; recalled entries surface as
background context.) Planning artifacts that pair with these live in `.omc/plans/` and
`.omc/specs/` (also committed); `QA_REPORT.md` records the last verification.
