"""Focused invariants for the generalized Reward Policy V2."""

from __future__ import annotations

from dataclasses import replace

import pytest

from pvzrl_rewards import (
    REWARD_COMPONENT_FIELDS,
    REWARD_POLICY_VERSION,
    RewardCompositionState,
    RewardConfig,
    compose_step_reward,
)


def observation(
    *,
    zombies: list[dict] | None = None,
    plants: list[dict] | None = None,
    kills: int = 0,
    wave: int = 1,
    mowers: int = 5,
    done: bool = False,
    terminal_hint: str = "running",
    loss_screen: bool = False,
    frame: int = 1,
) -> dict:
    return {
        "frameCount": frame,
        "rowCount": 5,
        "columnCount": 10,
        "killCount": kills,
        "wave": wave,
        "logicalMowerCount": mowers,
        "zombies": list(zombies or []),
        "zombieCount": len(zombies or []),
        "plants": list(plants or []),
        "plantCount": len(plants or []),
        "gameplayReady": not loss_screen,
        "done": done,
        "over": loss_screen,
        "terminalHint": terminal_hint,
        "onGameOverScreen": loss_screen,
        "gameOverTextVisible": loss_screen,
    }


def zombie(x: float, *, health: float = 100.0, maximum: float = 100.0, row: int = 0) -> dict:
    return {
        "row": row,
        "x": x,
        "alive": True,
        "health": health,
        "maxHealth": maximum,
        "type": 0,
    }


def step(
    previous: dict,
    current: dict,
    *,
    action_result: dict | None = None,
    state: RewardCompositionState | None = None,
    config: RewardConfig | None = None,
):
    return compose_step_reward(
        previous,
        current,
        action_result or {"decoded": {"kind": "wait"}},
        config=config or RewardConfig(),
        state=state or RewardCompositionState.initial(5),
    )


def test_policy_version_and_defaults() -> None:
    config = RewardConfig()
    assert REWARD_POLICY_VERSION == "generalized_threat_v2"
    assert config.win_reward == 15.0
    assert config.loss_penalty == 15.0
    assert config.wave_reward == 2.0
    assert config.kill_reward == 0.25
    assert config.threat_delta_coef == 0.75
    assert config.mower_loss_penalty == 2.0
    assert config.illegal_action_penalty == 0.1
    assert config.fusion_success_reward == 0.15


def test_terminal_wave_kill_and_illegal_signs() -> None:
    base = observation()
    win = step(base, observation(done=True, terminal_hint="possible_win", frame=2))
    loss = step(
        base,
        observation(
            done=True,
            terminal_hint="game_over_or_loss",
            loss_screen=True,
            frame=2,
        ),
    )
    progress = step(
        base,
        observation(kills=2, wave=3, frame=2),
        action_result={"decoded": {"kind": "plant"}, "illegalAction": True},
    )
    assert win.breakdown.component("win_loss_reward") == 15.0
    assert loss.breakdown.component("win_loss_reward") == -15.0
    assert progress.breakdown.component("kill_reward") == 0.5
    assert progress.breakdown.component("wave_reward") == 4.0
    assert progress.breakdown.component("illegal_penalty") == -0.1


@pytest.mark.parametrize(
    ("before_x", "after_x", "expected_sign"),
    [(8.0, 3.0, -1), (3.0, 8.0, 1), (5.0, 5.0, 0)],
)
def test_threat_direction(before_x: float, after_x: float, expected_sign: int) -> None:
    result = step(
        observation(zombies=[zombie(before_x)]),
        observation(zombies=[zombie(after_x)], frame=2),
    )
    value = result.breakdown.component("threat_delta_reward")
    assert (value > 0) - (value < 0) == expected_sign
    diagnostics = result.diagnostics_dict()
    assert diagnostics["threat_delta_reward"] == pytest.approx(value)
    assert -1.0 <= diagnostics["threat_clipped_delta"] <= 1.0


def test_damage_and_kill_reduce_threat() -> None:
    damaged = step(
        observation(zombies=[zombie(3.0, health=100)]),
        observation(zombies=[zombie(3.0, health=20)], frame=2),
    )
    killed = step(
        observation(zombies=[zombie(3.0)]),
        observation(zombies=[], kills=1, frame=2),
    )
    assert damaged.breakdown.component("threat_delta_reward") > 0.0
    assert killed.breakdown.component("threat_delta_reward") > 0.0
    assert killed.breakdown.component("kill_reward") == 0.25


def test_threat_reward_is_bounded_under_extreme_counts() -> None:
    far = [zombie(10.0, row=index % 5) for index in range(500)]
    close = [zombie(0.0, row=index % 5) for index in range(500)]
    toward = step(observation(zombies=far), observation(zombies=close, frame=2))
    away = step(observation(zombies=close), observation(zombies=far, frame=2))
    assert -0.75 <= toward.breakdown.component("threat_delta_reward") <= 0.0
    assert 0.0 <= away.breakdown.component("threat_delta_reward") <= 0.75


def test_plant_identity_health_and_position_do_not_shape_reward() -> None:
    previous = observation(
        zombies=[zombie(6.0)],
        plants=[{"row": 0, "column": 1, "type": 0, "health": 300, "maxHealth": 300}],
    )
    current = observation(
        zombies=[zombie(6.0)],
        plants=[{"row": 4, "column": 9, "type": 9999, "health": 1, "maxHealth": 300}],
        frame=2,
    )
    result = step(previous, current)
    assert result.breakdown.reward_total == pytest.approx(0.0)
    assert result.breakdown.component("plant_health_loss_penalty") == 0.0
    assert result.breakdown.component("role_positioning_reward") == 0.0


def test_mower_loss_is_penalized_once() -> None:
    first = step(observation(mowers=5), observation(mowers=4, frame=2))
    second = step(
        observation(mowers=4, frame=2),
        observation(mowers=4, frame=3),
        state=first.state,
    )
    third = step(
        observation(mowers=4, frame=3),
        observation(mowers=3, frame=4),
        state=second.state,
    )
    assert first.breakdown.component("mower_loss_penalty") == -2.0
    assert second.breakdown.component("mower_loss_penalty") == 0.0
    assert third.breakdown.component("mower_loss_penalty") == -2.0


def test_reward_total_is_exact_component_sum() -> None:
    result = step(
        observation(zombies=[zombie(7.0)], mowers=5),
        observation(zombies=[zombie(4.0)], kills=1, wave=2, mowers=4, frame=2),
    )
    payload = result.breakdown.to_dict()
    assert payload["reward_total"] == pytest.approx(
        sum(payload[name] for name in REWARD_COMPONENT_FIELDS)
    )


def test_legacy_strategy_coefficients_are_no_ops_even_if_nonzero() -> None:
    config = replace(
        RewardConfig(),
        role_positioning_reward=99.0,
        first_peashooter_in_row_reward=99.0,
        fusion_tier3_reward=99.0,
        fusion_threatened_row_bonus=99.0,
    )
    result = step(
        observation(),
        observation(plants=[{"row": 0, "column": 0, "type": 0}], frame=2),
        action_result={
            "decoded": {"kind": "plant", "plantType": 0, "row": 0, "column": 0},
            "plantPlaced": True,
        },
        config=config,
    )
    assert result.breakdown.reward_total == 0.0
    assert result.breakdown.component("role_positioning_reward") == 0.0
    assert result.breakdown.component("first_peashooter_in_row_reward") == 0.0
