"""Compatibility wrapper for the multiprocessing collector.

The persistent-worker implementation lives in ``mahjong.rl.trainer_mp``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from mahjong.engine import RiichiEngine
from mahjong.rl.trainer_mp import collect_parallel_batch_mp


def collect_batch_mp(
    engines: List[RiichiEngine],
    model: Any,
    cfg: Any,
    device: str,
    num_workers: Optional[int] = None,
) -> Tuple[Any, Dict[str, Any]]:
    return collect_parallel_batch_mp(engines, model, cfg, device, num_workers=num_workers)
