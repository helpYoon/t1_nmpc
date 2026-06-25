# tests/test_wb_gait.py
import numpy as np
from t1_nmpc.wb.gait_wb import FLY, RF, LF, STANCE, mode_to_stance, SLOW_WALK


def test_mode_enum_values():
    assert (FLY, RF, LF, STANCE) == (0, 1, 2, 3)


def test_mode_to_stance_table():           # [left_stance, right_stance]; RF=right-in-contact
    assert mode_to_stance(STANCE) == (True, True)
    assert mode_to_stance(RF) == (False, True)     # RF: right contact, LEFT swings
    assert mode_to_stance(LF) == (True, False)     # LF: left contact, RIGHT swings
    assert mode_to_stance(FLY) == (False, False)


def test_slow_walk_schedule():
    assert SLOW_WALK.duration == 1.7
    np.testing.assert_allclose(SLOW_WALK.event_phases, [0.65 / 1.7, 0.85 / 1.7, 1.5 / 1.7])
    np.testing.assert_array_equal(SLOW_WALK.mode_sequence, [LF, STANCE, RF, STANCE])


def test_mode_at_and_contact_flags():
    g = SLOW_WALK
    assert g.mode_at(0.3) == LF and g.contact_flags(0.3) == (True, False)   # right swings
    assert g.mode_at(0.75) == STANCE and g.contact_flags(0.75) == (True, True)
    assert g.mode_at(1.0) == RF and g.contact_flags(1.0) == (False, True)   # left swings
    assert g.mode_at(1.6) == STANCE
    assert g.mode_at(1.7 + 0.3) == LF        # wraps to phase 0.3


def test_phase_boundary_is_closed_open():
    # at an exact event phase the mode advances to the next (searchsorted side='right')
    assert SLOW_WALK.mode_at(0.65) == STANCE     # 0.65 is the LF->STANCE switch
