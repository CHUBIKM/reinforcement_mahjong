"""Compatibility wrapper for persistent-worker multiprocessing collection.

Use ``mahjong.rl.trainer_mp.collect_parallel_batch_mp`` for new code.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from mahjong.rl.trainer_mp import collect_parallel_batch_mp


def collect_batch_multiprocess(
    model: Any,
    cfg: Any,
    device: str,
    num_workers: Optional[int] = None,
) -> Tuple[Any, Dict[str, Any]]:
    return collect_parallel_batch_mp([], model, cfg, device, num_workers=num_workers)
