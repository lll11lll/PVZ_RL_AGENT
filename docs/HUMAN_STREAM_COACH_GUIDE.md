# Human and Crowd Coach Guide

> **Legacy local-coach subsystem:** This guide documents human coaching and the mock/local crowd-coach voting path. It is not Streamer Mode V1 and its commands, vote windows, raw mock-user fixtures, and PPO override semantics do not apply to V1. For live Twitch EventSub, strict FIFO commands, viewer hashing, intervention-aware PPO, BC, and autonomous checkpoint cycles, use [Streamer Mode V1](STREAMER_MODE.md). Streamer V1 refuses to run alongside these legacy action overrides.

PvZRL supports opt-in local coaching around the Adventure Generalist policy. This legacy coach subsystem has no hosted chat service or production Twitch/YouTube adapter; the separately maintained Streamer Mode V1 provides the read-only Twitch integration.

## Components

- `pvzrl_human_coach.py`: command parsing, validation, modes, decisions, reward shaping, and status.
- `pvzrl_stream_coach.py`: mock/local message ingestion, vote aggregation, anti-spam/rate limiting, and status.
- `pvzrl_assisted_coach.py`: moderated intervention records.
- `pvzrl_file_tail.py`: partial-record-safe local file tailing.
- `pvzrl_sb3.py`: arbitration before one policy step.
- `pvzrl_env.py`: shared cached legality and bridge execution.

Coaching does not change the 701-action / 4,297-observation policy contract.

## Command flow

```text
file / GUI queue / mock message
  -> parse command
  -> validate syntax and current frame
  -> map to Generalist action or bridge fusion request
  -> apply configured coach mode/precedence
  -> execute through the shared environment boundary
  -> record decision, outcome, reward, and status
```

Human coach takes precedence over crowd coach. When both are enabled, crowd input may be alive and parsed without being selected.

Crowd coach defaults to dry-run. Applying a crowd command requires explicit `--stream-coach-apply` and compatible mode configuration.

## Commands

```text
!plant <seed_index> <row> <column>
!fuse <seed_index> <row> <column>
!defend <row>
!economy
!wait
```

Examples:

```text
!plant 0 2 4
!fuse 1 1 0
!defend 3
!economy
!wait
```

- `!plant` selects the identity slot/cell action when legal.
- `!fuse` names the occupied source tile and ingredient slot. Validation probes bridge capability/candidates before selection. A ready equivalent duplicate identity slot may be chosen when the requested duplicate is unavailable.
- `!defend` chooses a legal defensive action in the requested lane.
- `!economy` chooses a legal economy-oriented action.
- `!wait` selects action `0`.

## Validation

Commands can be rejected or held pending for:

- malformed/unknown syntax;
- seed, row, or column bounds;
- inactive/unavailable/disabled/cooling slot;
- insufficient sun;
- action absent from the current cached mask;
- unavailable fusion support/probe;
- empty or incompatible source;
- bridge rejection;
- rate limit, vote threshold, stale record, or precedence.

All sources share the same current-frame `ActionDecisionCache`. Coaches do not maintain a separate legality table.

## Human coach modes

`--human-coach-command-mode` accepts:

- `override`: a valid different coach action replaces the policy action;
- `assist`: record/assist under the implemented hook semantics;
- `coach_only`: coach drives valid actions and safely waits when absent;
- `viewer_suggestion`: record a suggestion without forcing it.

Inspect current code/tests for exact selection behavior when changing these modes.

## Enable human coaching

Training example:

```powershell
python .\python\train_ppo.py `
  --config configs\ppo_adventure_generalist_14slot_identity_v1.json `
  --adventure-generalist-train `
  --run-dir runs\coach_generalist `
  --human-coach-enabled `
  --human-coach-command-path runs\coach_commands.txt `
  --human-coach-log-path runs\human_coach.jsonl `
  --human-coach-reward `
  --human-coach-fusion-enabled
```

Evaluation uses the same flags with `--adventure-generalist-eval` and a compatible `--model-path`.

Optional shaping flags include coach match, legal execution, override, fusion success, and tactical usefulness coefficients. Keep them small and verify exact reward-component totals.

## Enable mock crowd coaching

Mock script records are JSONL:

```json
{"t":5,"user":"mock_viewer_1","message":"!wait"}
{"t":8,"user":"mock_viewer_2","message":"!defend 2"}
```

Example dry-run training:

```powershell
python .\python\train_ppo.py `
  --config configs\ppo_adventure_generalist_14slot_identity_v1.json `
  --adventure-generalist-train `
  --run-dir runs\crowd_generalist `
  --stream-coach-enabled `
  --stream-coach-mode mock `
  --stream-coach-platform mock `
  --stream-coach-mock-script runs\mock_chat.jsonl `
  --stream-coach-log-path runs\stream_coach.jsonl `
  --stream-coach-dry-run
```

To apply validated selections, replace dry-run with explicit `--stream-coach-apply` and review rate/vote settings first.

`--stream-coach-command-path` tails appended local JSONL and starts at EOF to avoid replaying stale commands.

## GUI

The Coach tab appends local queue records and displays normalized human/crowd status. Train and eval controls launch Generalist commands with the selected coach configuration. The GUI does not bypass environment validation or call the bridge directly.

## Logs and status

Human JSONL events include matches, overrides, suggestions/rejections, pending decisions, and step outcomes. Useful fields:

- raw/parsed command and source;
- requested and selected slot/cell;
- policy/selected action;
- bridge command when fusion is used;
- validation legality/reason/diagnostics;
- duplicate-slot fallback evidence;
- execution result and reward components.

Crowd status includes messages seen, commands parsed/accepted/rejected, vote window/top commands, pending count, last rejection, selected command/action/votes, rate-limit health, match/override counts, and reward total.

Fusion probe diagnostics should retain candidate slots, readiness/cooldown/cost/sun, source identity, candidate match, bridge reason, and duplicate fallback.

## Troubleshooting

### Mock messages appear configured but never selected

Check whether human coach is enabled. Its precedence is intentional. Then inspect `runs/live_status.json`, command path, message/parse counters, pending commands, and crowd JSONL.

### Fusion remains pending

Distinguish transient insufficient-sun/cooldown from terminal bridge rejection. Inspect candidate slots and duplicate fallback. Do not repeatedly append the same command without understanding the pending reason.

### Command is legal in the UI but rejected at execution

Compare frame identity, cached mask decision, selected slot/source, and bridge rejection. Unity state can change between validation and execution.

### No log updates

Confirm resolved paths, parent directory permissions, source tail position, and that the coach mode is enabled in the child process—not only in GUI widgets.

## Verification

```powershell
python -m pytest -q python\test_human_coach.py python\test_stream_coach.py python\test_coach_file_tail.py python\test_gui_coach_queue.py
```

For live acceptance, retain a safe wait/plant command, a rejected command, a duplicate-slot fusion probe, precedence behavior, and bounded GUI/process shutdown. Do not expose private chat/profile contents in reports.
