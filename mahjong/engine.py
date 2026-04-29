from __future__ import annotations

from typing import Any, Callable
import importlib.util
import sys
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

_project_root = Path(__file__).parent.parent
_module_path = next(
    (_project_root / f"_mahjong_cpp{suffix}" for suffix in EXTENSION_SUFFIXES if (_project_root / f"_mahjong_cpp{suffix}").exists()),
    None,
)

if _module_path is not None:
    spec = importlib.util.spec_from_file_location("_mahjong_cpp", str(_module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load _mahjong_cpp extension from {_module_path}")
    _mahjong_cpp = importlib.util.module_from_spec(spec)
    sys.modules["_mahjong_cpp"] = _mahjong_cpp
    spec.loader.exec_module(_mahjong_cpp)
else:
    import _mahjong_cpp

from _mahjong_cpp import (  # type: ignore
    Action,
    ActionType,
    Phase,
    PlayerState,
    RiichiEngine as _CppRiichiEngine,
    RuleConfig,
    StepResult,
    analyze_yaku,
    count_yaochu_types,
    hand_to_str,
    is_agari,
    is_chiitoi,
    is_kokushi,
    is_standard_agari,
    is_tenpai,
    tile_to_str,
)
from mahjong.rules import RuleProfile

GameConfig = RuleProfile


class _SyncList(list):
    """Mutable list view that syncs all in-place edits back to C++ storage."""

    def __init__(self, data: list[Any], on_change: Callable[[list[Any]], None]):
        super().__init__(data)
        self._on_change = on_change

    def _commit(self) -> None:
        self._on_change(list(self))

    def __setitem__(self, key, value):  # type: ignore[override]
        super().__setitem__(key, value)
        self._commit()

    def __delitem__(self, key):  # type: ignore[override]
        super().__delitem__(key)
        self._commit()

    def append(self, value):  # type: ignore[override]
        super().append(value)
        self._commit()

    def extend(self, values):  # type: ignore[override]
        super().extend(values)
        self._commit()

    def insert(self, index, value):  # type: ignore[override]
        super().insert(index, value)
        self._commit()

    def pop(self, index=-1):  # type: ignore[override]
        out = super().pop(index)
        self._commit()
        return out

    def remove(self, value):  # type: ignore[override]
        super().remove(value)
        self._commit()

    def clear(self):  # type: ignore[override]
        super().clear()
        self._commit()

    def sort(self, *args, **kwargs):  # type: ignore[override]
        super().sort(*args, **kwargs)
        self._commit()

    def reverse(self):  # type: ignore[override]
        super().reverse()
        self._commit()


class _PlayerStateProxy:
    def __init__(self, impl: PlayerState):
        object.__setattr__(self, "_impl", impl)

    def __getattr__(self, name: str):
        impl = object.__getattribute__(self, "_impl")
        if name == "hand34":
            return _SyncList(list(impl.hand34), lambda v: setattr(impl, "hand34", v))
        if name == "river":
            return _SyncList(list(impl.river), lambda v: setattr(impl, "river", v))
        if name == "melds":
            return _SyncList(list(impl.melds), lambda v: setattr(impl, "melds", v))
        return getattr(impl, name)

    def __setattr__(self, name: str, value):
        impl = object.__getattribute__(self, "_impl")
        setattr(impl, name, value)


class RiichiEngine:
    """Thin Python compatibility wrapper over the C++ core engine."""

    def __init__(self, seed: int = 0, config: RuleProfile | None = None):
        object.__setattr__(self, "_impl", _CppRiichiEngine(seed=seed, config=config))

    def __getattr__(self, name: str):
        if name == "players":
            return [_PlayerStateProxy(p) for p in self._impl.players]
        return getattr(self._impl, name)

    def __setattr__(self, name: str, value):
        if name == "_impl":
            object.__setattr__(self, name, value)
            return
        setattr(self._impl, name, value)

    def reset(self, dealer: int = 0) -> None:
        self._impl.reset(dealer=dealer)

    def draw(self) -> int:
        return int(self._impl.draw())

    def legal_discard_mask(self):
        return self._impl.legal_discard_mask()

    def legal_discards(self):
        return self._impl.legal_discards()

    def legal_actions(self):
        return self._impl.legal_actions()

    def discard(self, tile: int) -> None:
        self._impl.discard(tile)

    def apply_action(self, action: Action) -> StepResult:
        return self._impl.apply_action(action)

    def step(self, discard_tile: int) -> StepResult:
        return self._impl.step(discard_tile)

    def get_obs(self, seat: int | None = None):
        return self._impl.get_obs(seat=seat)

    def validate_invariants(self) -> None:
        self._impl.validate_invariants()

    def export_replay(self):
        return self._impl.export_replay()

    def play_random(self, max_steps: int = 20000, verbose: bool = False) -> StepResult:
        return self._impl.play_random(max_steps=max_steps, verbose=verbose)

    def _should_abort_suufon_renda(self) -> bool:
        return bool(self._impl._should_abort_suufon_renda())

    def _yaku_info_for_win(self, winner: int, win_type: str, winning_hand34: list[int], win_tile: int | None):
        return self._impl._yaku_info_for_win(winner, win_type, winning_hand34, win_tile)


__all__ = [
    "Action",
    "ActionType",
    "Phase",
    "PlayerState",
    "RiichiEngine",
    "RuleConfig",
    "StepResult",
    "GameConfig",
    "tile_to_str",
    "hand_to_str",
    "is_kokushi",
    "is_chiitoi",
    "is_standard_agari",
    "is_agari",
    "is_tenpai",
    "count_yaochu_types",
    "analyze_yaku",
]


if __name__ == "__main__":
    eng = RiichiEngine(seed=42, config=GameConfig())
    eng.reset(dealer=0)

    print("=== 初始牌局（仅自测展示，真实对局中他家手牌不可见）===")
    for p in eng.players:
        print(f"玩家{p.seat} 手牌：{hand_to_str(p.hand34)}")

    res = eng.play_random(verbose=False)

    print("\n=== 对局结果 ===")
    print(res)

    for p in eng.players:
        river_str = " ".join(tile_to_str(t) for t in p.river)
        print(f"玩家{p.seat} 河：{river_str}")
