# Task: Rewrite COMPSCI 773 lecture slides into an exam-ready Obsidian reference library

## Context
Convert the disorganised lecture slides for **COMPSCI 773 — Intelligent Vision Systems** (University of Auckland; maths-heavy computer vision) into a clean, concise Obsidian reference library. The slides in `@slides` are the source of truth for what the course covers — don't assume a topic list. Reader = a student revising for a **short-answer, open-book exam**. Optimise for accuracy, conciseness, navigability, exam usefulness.

**Source hierarchy.** `@slides` is primary. `@COMPSC773.pdf` is a secondary reference — use it to clarify, fill gaps, reconcile confusing slides, or add structure, but never let it override the slides (some of it is outdated). When it informs content not clearly in the slides, flag with `> [!note]`. On conflict, prefer the slides and note the discrepancy.

**Recency.** Filenames encode it: `W<week><L|T><n>` (week number; Lecture or Tutorial; sequence). More recent = higher week; within a week, higher `n`, with tutorials following that week's lectures. When sources state a fact differently, the most recent wins — add a brief `> [!note]` saying what changed and which source superseded which. (Slides still beat the supplement regardless of recency.)

## Conventions (every file)
- **Obsidian Markdown:** inline `$...$`, block `$$...$$`, wikilinks `[[topic-name]]`, `> [!note]`/`> [!warning]`/`> [!example]` callouts.
- **File names:** kebab-case, e.g. `topics/epipolar-geometry.md`.
- **Notation — consistent across ALL files.** Fix one convention, reuse everywhere:
  - scalars italic ($f$, $\lambda$), vectors bold lowercase ($\mathbf{p}$), matrices bold uppercase ($\mathbf{M}$), sets/frames calligraphic where helpful.
  - state coordinate systems/frames explicitly (homogeneous vs. inhomogeneous; which frame a quantity lives in).
  - bind to the shared symbol glossary (see O2). When a slide's notation differs, normalise it and record the original in a `> [!note]`.
- **Style: terse, information-dense.** Cut filler ("note that", "essentially", "in order to", "simply"). Favour fragments, symbols, math over sentences where meaning holds. Every line should be revision-worthy.
- **Proofs:** keep them, but put them in the appendix (A7), not in the steps.

## Per-topic file structure (`@topics/<topic>.md`)
Start with a TOC linking to each section below (and to named variants under A6, proofs under A7). Then, in order:

**A1. Purpose** — problem it solves and when to reach for it (2–4 sentences).

**A2. Overview** — steps as a short ordered list, *no math*. Include **key definitions** (terms, quantities, named matrices/constraints), one line each.

**A3. Strengths, shortcomings & limitations** — bulleted. Then **"Likely exam questions"**: reproduce the exam/practice questions that appear in the slides (normalised to convention; keep any answer the slides give, concise), then bullet the **likely variations** of those questions.

**A4. Directions of reasoning** — examine the algorithm both ways:
- **forward** (inputs → output, A→B), and
- any **reverse/inferential** direction it's examined in (given output/intermediate state, recover inputs or justify a step, B→A).
For each: what's given, what's asked, the key inference/manipulation.

**A5. Standard implementation**
- **a. Setup** — parameters, notation, assumptions, I/O.
- **b. Steps** — ordered list, one step per item, each with its math in LaTeX; the specifics, not hand-waving. No proofs here — cross-reference A7 where a step rests on one.

**A6. Variations** — named variants, each with the specific change to setup/steps and when it's used. Keep all variants of a topic in this file. Link related variants/topics with `[[ ]]`.

**A7. Appendix — Proofs & derivations** — *only if the slides contain any.* One subheading per proof/derivation; capture the steps, and link from the relevant A5b step.

## Pipeline (Claude Code)

> [!important] Context budget
> Some decks are ~80 slides — too large for one subagent. No agent ever holds a whole large deck: O1 works text-first and chunks big decks, the orchestrator sees only summaries + manifests, O3 retrieves exact slides on demand.

**O0 — Prep & chunk.** Inventory `@slides`. Per PDF, extract per-page **text** first (cheap); flag for **rasterisation** only pages that need vision — diagrams, figures, equations that don't extract cleanly (render at modest DPI, grayscale). Threshold ~25 slides: process smaller decks whole; pre-split larger ones into ~15–20-slide windows (`lectureN.part1`, …) and run chunks in parallel. *(Default. Alternative: each O1 agent chunks its own deck sequentially — fewer files, slower.)*

**O1 — Per-slide summaries.** Spawn one subagent per deck or chunk (Task tool), in parallel; each works text-first (pulling only flagged pages as images) and writes a chunk summary. Multi-chunk decks get a short merge step → `@outputs/lecture_summaries/<lecture>.md`. Each summary captures: algorithms/procedures, key concepts/definitions, notation **verbatim** (so O2/O3 can reconcile), the filename's `W<week><L|T><n>` marker (for recency), anything exam-relevant, and a **slide manifest** table: `slide # → topic/algorithm → has-key-equation? → has-figure?`. **Factual summaries only — no synthesis.**

**O2 — Index + shared conventions (fix before fanout).** Read only `@outputs/lecture_summaries` (summaries + manifests, not the decks). Produce:
- `@outputs/topic_index.md`: deduplicated **canonical topics**, each with lecture(s) + **slide ranges** (from manifests). Merge near-duplicates (same algorithm, different names/notation) into one canonical topic with aliases; set boundaries (one file per algorithm/procedure, variants folded into A6). Fix the **canonical slug** for each topic — O3 links only to these.
- `@outputs/symbol_glossary.md`: concept → canonical symbol → frame/coordinate convention, covering recurring quantities (camera/projection matrix, fundamental/essential matrix, rotation, intrinsics, etc.). All O3 agents bind to it.

**O3 — Per-topic synthesis.** For each canonical topic, spawn a subagent (Task tool) to produce A1–A7. It reads the summaries + manifest, then uses the slide ranges to **retrieve only the exact pages it needs** (text, plus flagged figures) for the precise math in A5b and any A7 proof — never a whole deck. Draw on **every** lecture where the topic appears; reconcile conflicting/partial/differently-notated coverage into one treatment, applying recency. Where slides are confusing or incomplete, consult `@COMPSC773.pdf` as secondary (flag supplement-derived content). Link only to canonical slugs from `topic_index.md`; bind to `symbol_glossary.md`. Output `@topics/<topic>.md`.

**O4 — Taxonomy.** Write `@topics/README.md`: how every topic links to the others (prerequisites, specialisations, shared subroutines, "X used inside Y"). Include a short table and a **Mermaid** graph, with `[[ ]]` links.

**O5 — Verify (last).** Confirm: every `topic_index.md` topic has a file; every file has the TOC + A1–A6 in order (A7 present iff the topic has proofs); notation matches the glossary across files; every `[[ ]]` and intra-doc link resolves (links use only canonical slugs). Report gaps or unresolved conflicts — don't silently fill them.

## Guardrails
- Conceptual/theoretical topic with no step-based algorithm: still make a file but adapt — keep A1–A3; under A5 give definitions/relationships instead of steps (skip A5b if there are genuinely none).
- Slides are the source of truth. To resolve an ambiguity beyond them, flag in a `> [!warning]` rather than presenting as course content.