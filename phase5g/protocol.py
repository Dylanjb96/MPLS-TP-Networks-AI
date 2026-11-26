"""Strict disjoint seed roles. No Phase 5G final hold-out access is implemented."""
from __future__ import annotations

G_DEV_TOPOLOGY = range(501, 521)
G_DEV_STREAM = range(74000, 74020)
G_TRAIN_TOPOLOGY = range(601, 631)
G_TRAIN_STREAM = range(75000, 75020)
G_VALIDATION_TOPOLOGY = range(701, 711)
G_VALIDATION_STREAM = range(76000, 76010)
G_RESERVED_FINAL_TOPOLOGY = range(801, 821)
G_RESERVED_FINAL_STREAM = range(77000, 77003)
F_RESERVED_FINAL_TOPOLOGY = range(401, 421)
F_RESERVED_FINAL_STREAM = range(73000, 73003)


def check_seed_role(role: str, topology: int, stream: int) -> None:
    allowed = {
        'dev': (G_DEV_TOPOLOGY,G_DEV_STREAM),
        'train': (G_TRAIN_TOPOLOGY,G_TRAIN_STREAM),
        'validation': (G_VALIDATION_TOPOLOGY,G_VALIDATION_STREAM),
    }
    if role not in allowed:
        raise ValueError('Phase 5G final seeds are inaccessible through development tools')
    t,s = allowed[role]
    if int(topology) not in t or int(stream) not in s:
        raise ValueError(f'{role}: topology or demand seed outside {role} partition')
    if int(topology) in F_RESERVED_FINAL_TOPOLOGY or int(stream) in F_RESERVED_FINAL_STREAM:
        raise ValueError('Phase 5F Stage 5 seed reuse forbidden')
