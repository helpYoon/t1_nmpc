"""Aligator-specific MPC settings (validated operating point: N=20, maxit=2, 4 threads -> 12ms)."""
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class AligatorConfig:
    N: int = 20                 # horizon nodes
    max_iters: int = 2          # ProxDDP inner-iteration budget (validated Phase-1 RT point: ~12ms)
    max_iters_transition: int = 5  # extra budget when contact flags change at a tick (variable budget)
    max_al_iters: int = 2       # outer AL iterations
    mu_init: float = 1e-2       # AL penalty
    tol: float = 1e-3
    num_threads: int = 4
    hard_cones: bool = True     # True: NegativeOrthant; False: RelaxedLogBarrierCost (OCS2-faithful)
    hard_swing_z: bool = True    # swing-z as a HARD z-only equality (OCS2-faithful); soft cost can't lift the foot
    FS: int = 6                 # 6D contact wrench per foot
    cone_eps: float = 1e-3      # friction-cone relaxation arg
    barrier_thr: float = 0.1    # RelaxedLogBarrierCost threshold (soft mode)
    # cost weights (mapped from crocoddyl Q/R intent)
    # Lateral balance in the tested upstream (wb_humanoid_mpc) is EMERGENT from the friction/CoP/contact
    # constraints, NOT a sway reference -> the horizontal base/CoM must be free to shift over the stance
    # foot. Keep base-xy weight LOW (don't pull the CoM back to a centered stand) but height + orientation firm.
    w_base_xy: float = 20.0     # horizontal base position: tracks the lateral-transfer reference (base-y
                                # shifted over the stance foot in single support -> the CoM weight shift)
    w_base_z: float = 50.0      # base height: firm
    w_base_ori: float = 50.0    # base orientation: firm
    w_joint_pos: float = 5.0
    w_vel: float = 1.0
    w_force_reg: float = 1e-3
    w_accel_reg: float = 1e-2
    w_swing_z: float = 1000.0   # must dominate joint-reg or the foot won't lift (OCS2 makes swing-z HARD)
    w_swing_force: float = 1e2  # heavy reg pinning an inactive foot's force slots to zero
    w_term_scale: float = 5.0

def make_aligator_config() -> AligatorConfig:
    return AligatorConfig()
