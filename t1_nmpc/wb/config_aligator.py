"""Aligator-specific MPC settings (validated operating point: N=20, maxit=2, 4 threads -> 12ms)."""
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class AligatorConfig:
    N: int = 20                 # horizon nodes
    max_iters: int = 2          # ProxDDP inner-iteration budget (raise to 3-5 at transitions in Phase 2)
    max_al_iters: int = 2       # outer AL iterations
    mu_init: float = 1e-2       # AL penalty
    tol: float = 1e-3
    num_threads: int = 4
    hard_cones: bool = True     # True: NegativeOrthant; False: RelaxedLogBarrierCost (OCS2-faithful)
    FS: int = 6                 # 6D contact wrench per foot
    cone_eps: float = 1e-3      # friction-cone relaxation arg
    barrier_thr: float = 0.1    # RelaxedLogBarrierCost threshold (soft mode)
    # cost weights (mapped from crocoddyl Q/R intent)
    w_base_pose: float = 50.0
    w_joint_pos: float = 5.0
    w_vel: float = 1.0
    w_force_reg: float = 1e-3
    w_accel_reg: float = 1e-2
    w_swing_z: float = 100.0
    w_swing_force: float = 1e2  # heavy reg pinning an inactive foot's force slots to zero
    w_term_scale: float = 5.0

def make_aligator_config() -> AligatorConfig:
    return AligatorConfig()
