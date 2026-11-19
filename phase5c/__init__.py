"""Phase 5C: on-policy masked PPO and discrete offline BCQ/CQL/AWAC/IQL.

Research adaptation for the existing Phase 3 protected-LSP scenario surrogate.
Does not implement RFC 8227 MSRP signaling or switching procedures.
"""
VERSION = 'PHASE5C_DISCRETE_RL_V1'
ALGORITHMS = ('ppo', 'bcq', 'cql', 'awac', 'iql')
