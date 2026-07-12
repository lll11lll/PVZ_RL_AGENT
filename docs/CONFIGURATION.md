# PvZRL Configuration Resolution

PvZRL resolves training, evaluation, Adventure, coach, diagnostics, artifact, and bridge settings with one precedence rule:

```text
explicit CLI value > JSON configuration > mode-specific default > global default
```

`python/pvzrl_config.py` defines the immutable `ResolvedRunConfig` view and its typed sections. `train_ppo.build_resolved_config()` constructs that view; `train_ppo.build_config()` remains the compatibility adapter that returns the historical flat dictionary consumed by existing runtime code and written to `resolved_config.json`.

The typed view also exposes an immutable `value_sources` mapping for resolved fields, using `cli`, `json`, `mode_default`, or `global_default`. This makes parser-default mistakes inspectable without adding provenance keys to the legacy flat output.

## What counts as an explicit CLI value

An argparse value of `None` means the option was not supplied. Falsey values that can be supplied explicitly, including `0`, `False`, and an empty string, are still explicit and win over JSON. JSON keys are authoritative when present, including an explicit JSON `null`; downstream type validation continues to determine whether that value is valid for the selected setting.

Precedence-sensitive Adventure options use `None` as their parser default:

- `--advance-on-wins`
- `--max-adventure-levels`
- `--max-attempts-per-level`
- `--adventure-start-level`

This prevents the parser defaults from silently replacing values in a JSON file. Their global defaults remain `1`, `5`, `10`, and `1`, respectively.

Boolean enable-only switches retain their historical behavior: an asserted CLI switch enables the feature; otherwise the JSON value or existing default is used. Paired enable/disable options already use an unset state where a three-way choice is required.

Run-mode shortcut flags are resolved as one semantic field before any mode-specific defaults are applied. Explicit `--run-mode`, Adventure Generalist, Level 3, Adventure Eval, and generic train/eval selectors therefore cannot be overridden by a JSON `run_mode`. Conflicting explicit specialized selectors fail before startup. Runtime dispatch uses the resolved mode, so a redundant generic `--eval` cannot route an Adventure request into fixed evaluation.

`stream_coach_mode` and `stream_coach_platform` are aliases for one semantic value. All explicit CLI aliases are considered before either JSON alias, and both compatibility output keys are kept coherent.

## Mode-specific defaults

Mode defaults apply only when neither CLI nor JSON supplied the value. Current examples include:

- quick-wait board and gameplay-ready timeouts;
- the Level 3 specialist target and four-card seed list;
- the Adventure Generalist 14-slot action capacity and identity loadout;
- Adventure Generalist unlock-capacity inference.

Explicit `plant_types` are resolved through the same CLI-over-JSON rule. Their exact slot order must match the canonical IDs resolved from `seed_list`, including duplicate slots; a mismatch blocks startup instead of silently changing decoder semantics.

The typed view groups the resolved values into optimization, environment, seed/action, Adventure, fusion, coach, diagnostics, artifacts, bridge, model-contract, and reward sections. GUI widget state remains owned by `pvzrl_gui.py`; values the GUI launches are serialized as ordinary CLI options and enter the same resolver.

## Compatibility guarantees

- Existing CLI flag names and JSON keys remain accepted.
- The flat resolved output retains its existing keys and JSON-compatible value shapes.
- Runtime consumers may migrate section by section while `build_config()` provides the compatibility window.
- Model metadata validation checks the resolved decoder, action count, slot identity, board geometry, placement range, and actual loaded observation shape.

Recognized legacy no-op fields emit `IgnoredLegacyConfigWarning` instead of disappearing silently. This currently covers top-level `enable_fusion_diagnostics` and the unused `proximity_penalty` reward field in both top-level and nested `reward` JSON shapes. `proximity_penalty` remains in the resolved reward dictionary during the compatibility window even though no runtime reward term consumes it.

The precedence matrix, typed round-trip, parser defaults, mode defaults, schedule paths, and falsey-value behavior are locked by `python/test_resolved_config.py`.
