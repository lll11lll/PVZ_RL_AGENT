# Adventure Generalist Coach Quickstart

> **Legacy local coach quickstart:** The **Stream Coach** below is the mock/local crowd-voting coach. It is not Streamer Mode V1, its syntax and vote aggregation do not describe Twitch FIFO control, and it cannot be enabled at the same time as V1. For Twitch EventSub setup, one-based V1 commands, PPO + BC, protected checkpoint cycles, and endurance testing, use [Streamer Mode V1](STREAMER_MODE.md).

## GUI

1. Open `python/pvzrl_gui.py`.
2. In the **Adventure Generalist** tab, configure training or eval as usual.
3. In the **Coach** tab:
   - Enable **Human Coach** and/or **Stream Coach**.
   - Keep queue/log defaults unless you need custom paths.
   - Optional: enable fusion planning/bridge flags.
4. Launch **Start Adventure Generalist Train** or **Start Adventure Generalist Eval**.
5. Use the coach input box to send commands like:
   - `plant 0 2 4`
   - `plant(0,2,4)`
   - `!plant 0 2 4`
   - `fuse 0 1 1`
   - `wait`
   - `defend 3`
   - `economy`

## CLI

Example train:

```bash
python python/train_ppo.py --adventure-generalist-train \
  --run-dir runs/generalist_with_coach \
  --human-coach-enabled \
  --human-coach-command-path runs/coach_commands.jsonl \
  --human-coach-log-path runs/human_coach.jsonl \
  --human-coach-reward \
  --stream-coach-enabled \
  --stream-coach-platform mock \
  --stream-coach-command-path runs/coach_commands.jsonl \
  --stream-coach-log-path runs/stream_coach.jsonl \
  --stream-coach-reward \
  --coach-allow-fusion-planning \
  --fusion-bridge-enabled \
  --live-status-path runs/live_status.json
```

Example eval:

```bash
python python/train_ppo.py --adventure-generalist-eval \
  --model-path runs/generalist_with_coach/model.zip \
  --human-coach-enabled \
  --human-coach-command-path runs/coach_commands.jsonl \
  --stream-coach-enabled \
  --stream-coach-mode mock \
  --stream-coach-platform mock \
  --stream-coach-dry-run \
  --stream-coach-min-votes 1 \
  --stream-coach-mock-script scripts/mock_stream_commands.jsonl \
  --live-status-path runs/live_status.json
```

Mock stream smoke test:

```bash
python python/train_ppo.py --adventure-generalist-eval \
  --model-path runs/generalist_with_coach/model.zip \
  --auto-select-seeds \
  --stream-coach-enabled \
  --stream-coach-mode mock \
  --stream-coach-platform mock \
  --stream-coach-dry-run \
  --stream-coach-min-votes 1 \
  --stream-coach-mock-script scripts/mock_stream_commands.jsonl \
  --max-steps 1000 \
  --game-speed 5 \
  --live-status-path runs/live_status.json
```

Dry-run is the default for stream coach testing. It drains mock messages, parses, validates, aggregates, and writes diagnostics without changing the PPO/env action. Use `--stream-coach-apply` only when you want validated safe stream commands to influence the active coach path.

The mock script JSONL format is raw chat text by step:

```jsonl
{"t": 5, "user": "mock_viewer_1", "message": "!defend 2"}
{"t": 10, "user": "mock_viewer_2", "message": "!economy"}
{"t": 15, "user": "mock_viewer_3", "message": "!wait"}
{"t": 20, "user": "mock_viewer_4", "message": "!plant 1 2 4"}
{"t": 25, "user": "mock_viewer_5", "message": "!prefer Peashooter"}
```

Current parser syntax is the same as human coach syntax: `!plant <seed> <row> <col>`, `!fuse <seed> <row> <col>`, `!defend <row>`, `!economy`, and `!wait`. Unsupported chat commands are rejected with reasons and appear in live diagnostics.

## Diagnostics

- Live status file: `runs/live_status.json`
- Human coach log: `runs/human_coach.jsonl`
- Stream coach log: `runs/stream_coach.jsonl`
- GUI shows:
  - mode/action_count/decoder/current level/wave/sun
  - stream enabled/mode/alive, last message, last parsed/applied command
  - dry-run/apply status and stale startup clearing
  - accepted/rejected counts, last reject reason, pending command count
  - human + stream last command/action/votes
  - override/match/reject counters
  - fusion bridge availability and last fusion result

If GUI stream diagnostics do not update, confirm the GUI and run both use `--live-status-path runs/live_status.json`, check `coach.stream_coach_*` plus flat `stream_coach_*` fields in the file, and inspect `last_stream_reject_reason` for parser mismatches.
