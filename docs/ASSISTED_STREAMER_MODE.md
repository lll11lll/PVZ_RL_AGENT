# Assisted Streamer Mode

> **Legacy moderated-coach UI:** Despite its historical name, this document describes the local Tk assisted-coach queue, not Streamer Mode V1. It does not use Twitch EventSub, the V1 FIFO/TTL controller, intervention-aware PPO, behavior cloning, or baseline/current/best cycles. See [Streamer Mode V1](STREAMER_MODE.md) for the maintained Twitch training workflow. Streamer V1 and these coach action overrides are mutually exclusive.

Assisted Streamer Mode turns the Tkinter dashboard into a moderated human-in-the-loop control surface. The PPO agent can keep training or evaluating while local coaches and future viewer adapters submit typed commands.

## Local workflow

1. Open `python/pvzrl_gui.py` and select **Assisted** or **Fusion** in the Train/Eval lab-mode selector.
2. In **Coach**, build a command and select **Send Structured Command**. This adds a validated `pending` item; it does not immediately affect the game.
3. Select the queue row, then **Approve**, **Reject**, or change the builder fields and choose **Modify**.
4. Choose **Execute** for an approved command. The dashboard writes it to the existing `runs/coach_commands.jsonl` queue, where the existing coach parser and live-board validator make the final legality decision.

Raw manual input remains available for parser-native commands and bypasses dashboard moderation.

## Command and mode model

The typed schema is implemented in `python/pvzrl_assisted_coach.py`. Commands have a stable ID, timestamp, source/user, row, column, target, and one of four states: `pending`, `approved`, `rejected`, or `executed`.

- `PLANT` and `FUSE` require row `0-4`, column `0-8`, and seed slot `0-13`.
- `REMOVE` requires a row and column.
- `BOOST` requires row, column, and a target.
- `SAVE_SUN`, `PAUSE_AGENT`, `RESUME_AGENT`, and `FORCE_EVAL` have no board coordinates.
- `PLANT`, `FUSE`, and `SAVE_SUN` currently have safe adapters to the established coach parser. Other commands remain visible, valid moderation records but are rejected at Execute with an explicit backend-adapter message. This prevents a future-facing command from being mistaken for a different game action.

Execution modes:

- `override`: a legal approved command replaces the PPO action.
- `assist`: validates and logs the command as a suggestion while the PPO action executes.
- `coach_only`: waits when no approved coach command is available.
- `viewer_suggestion`: viewer-origin commands follow suggestion semantics and must pass dashboard approval.

Fusion commands are first-class typed commands. The dashboard validates their shape, then the existing fusion probe validates bridge availability, source/target legality, cooldown, sun, and live board state at runtime.

## Intervention data

Unified JSONL records default to `logs/interventions/interventions.jsonl` for runtime events and `logs/interventions/dashboard_interventions.jsonl` for moderation events. Each record carries run/episode/step identity, train/eval mode, model action, human command, source, board summary when available, reward fields, status, and mode metadata. These append-only records are intended for later intervention analysis or training-data extraction.

The existing stream mock JSONL source remains supported. Queue payloads may include `parser_command`, allowing the display-oriented `COMMAND ROW COL TARGET` text to coexist with the established parser's seed-first serialization.
