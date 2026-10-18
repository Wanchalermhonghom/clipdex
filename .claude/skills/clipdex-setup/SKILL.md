---
name: clipdex-setup
description: Set up clipdex on this computer step by step - check tools, pick a hardware profile, download models, test-index a few files, measure speed and memory, calibrate the search thresholds and connect the MCP server. Use when the user wants to install, set up, configure, tune or adapt clipdex for their machine.
argument-hint: "[library folder, e.g. /Volumes/T7]"
---

Set up clipdex for the library folder: $ARGUMENTS

If no folder is given, ask for it first.

Follow the section "Procedure: adapt clipdex to this machine" in AGENTS.md, one step at a time:

1. Run each command yourself with `uv run clipdex ...` from the repository root.
2. After each step, show the user the important numbers in two or three short lines.
3. Ask before you download models and before you overwrite an existing `config.toml`.
4. Never start the full index yourself. It can run for hours. Give the user the command for their own terminal.
5. If a step fails, read the error, fix the cause, and run the step again. Do not skip steps.
6. Finish with the report from step 7.
