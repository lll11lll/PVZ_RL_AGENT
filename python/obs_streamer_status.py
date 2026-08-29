import json
import time
from pathlib import Path

STATUS_PATH = Path(
    r"runs\streamer_v1\public_alpha_01\live_status.json"
)

OUTPUT_PATH = Path(
    r"runs\streamer_v1\public_alpha_01\obs_status.txt"
)


def format_last_command(data):
    """
    Build a readable representation of the last parsed/executed
    viewer action from live_status.json.
    """
    action = data.get("last_viewer_action") or {}

    if not action:
        return "None yet"

    command_type = action.get("command_type", "?")

    # These values in live_status have previously been zero-based internally.
    # Add 1 here so OBS shows the same coordinates/slots viewers type.
    slot = action.get("requested_slot")
    row = action.get("requested_row")
    col = action.get("requested_col")

    if command_type == "slot":
        if slot is not None and row is not None and col is not None:
            return f"!slot {slot + 1} {row + 1} {col + 1}"

    if command_type == "plant":
        plant = action.get("requested_plant") or action.get("plant_name", "?")

        if row is not None and col is not None:
            return f"!plant {plant} {row + 1} {col + 1}"

    if command_type == "fuse":
        if row is not None and col is not None:
            return f"!fuse {row + 1} {col + 1}"

    return str(command_type)


def format_result(data):
    action = data.get("last_viewer_action") or {}

    if not action:
        return "WAITING FOR CHAT"

    status = action.get("execution_status", "")
    reject_reason = action.get("bc_demo_reject_reason", "")

    if status == "executed_verified":
        result = action.get("resulting_plant", "")
        if result:
            return f"EXECUTED -> {result}"
        return "EXECUTED"

    if status:
        return status.upper()

    if reject_reason:
        return f"REJECTED: {reject_reason}"

    return "UNKNOWN"


while True:
    try:
        data = json.loads(
            STATUS_PATH.read_text(encoding="utf-8")
        )

        phase = data.get("streamer_mode", "UNKNOWN")
        level = data.get("current_level", "?")
        twitch = data.get("twitch_connection_state", "unknown")

        # -------------------------
        # Command counters
        # -------------------------

        received = data.get(
            "twitch_notifications_received",
            0
        )

        queue = data.get("streamer_command_queue") or {}
        counters = queue.get("counters") or {}

        accepted = counters.get("accepted", 0)
        executed = counters.get("executed", 0)
        blocked = counters.get("temporarily_blocked", 0)
        rejected = counters.get("permanently_rejected", 0)
        expired = counters.get("expired", 0)

        # Viewer interventions that actually entered the environment.
        interventions = data.get(
            "viewer_intervention_count",
            executed
        )

        demos = data.get(
            "bc_demonstration_count",
            0
        )

        bc_updates = data.get(
            "bc_update_count",
            0
        )

        # -------------------------
        # Last command
        # -------------------------

        last_command = format_last_command(data)
        last_result = format_result(data)

        # -------------------------
        # Pending command
        # -------------------------

        pending = data.get("pending_viewer_command")
        pending_reason = data.get(
            "pending_viewer_block_reason",
            ""
        )

        reserved_sun = data.get(
            "streamer_reserved_sun",
            0
        )

        if pending:
            pending_text = (
                f"WAITING: {pending_reason or 'temporarily blocked'}"
            )

            if reserved_sun:
                pending_text += f" | Reserved Sun: {reserved_sun}"
        else:
            pending_text = "None"

        # -------------------------
        # Phase display
        # -------------------------

        if phase == "STREAM_TRAIN":
            phase_text = "CHAT TRAINING ACTIVE"
            control_text = "CHAT CONTROL: ENABLED"

        elif phase == "EVALUATE":
            phase_text = "AUTONOMOUS EVALUATION"
            control_text = "CHAT CONTROL: DISABLED"

        else:
            phase_text = phase
            control_text = "CHAT CONTROL: WAITING"

        if twitch != "connected":
            twitch_text = f"TWITCH: {twitch.upper()}"
        else:
            twitch_text = "TWITCH: CONNECTED"

        # -------------------------
        # OBS output
        # -------------------------

        display = (
            f"RLPVZ - HUMAN ASSISTED RL\n"
            f"{phase_text} | Level 1-{level}\n"
            f"{control_text} | {twitch_text}\n"
            f"\n"
            f"LAST COMMAND: {last_command}\n"
            f"RESULT: {last_result}\n"
            f"PENDING: {pending_text}\n"
            f"\n"
            f"CHAT RECEIVED: {received}\n"
            f"COMMANDS ACCEPTED: {accepted}\n"
            f"AI INTERVENTIONS: {interventions}\n"
            f"BLOCKED: {blocked} | REJECTED: {rejected} | EXPIRED: {expired}\n"
            f"BC DEMOS: {demos} | BC UPDATES: {bc_updates}"
        )

        OUTPUT_PATH.write_text(
            display,
            encoding="utf-8"
        )

    except Exception as e:
        OUTPUT_PATH.write_text(
            f"RLPVZ\nSTATUS UNAVAILABLE\n{type(e).__name__}",
            encoding="utf-8"
        )

    time.sleep(0.5)