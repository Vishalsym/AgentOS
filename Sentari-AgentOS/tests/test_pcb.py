import pytest

from sentari.pcb import AgentPCB, AgentState


def test_new_pcb_starts_in_new_state():
    pcb = AgentPCB.new(priority=5, quota_total=10)
    assert pcb.state is AgentState.NEW
    assert pcb.quota_used == 0
    assert pcb.quota_remaining == 10


def test_quota_remaining_never_negative():
    pcb = AgentPCB.new(priority=5, quota_total=10)
    pcb.quota_used = 15
    assert pcb.quota_remaining == 0


def test_valid_transition_chain():
    pcb = AgentPCB.new(priority=5, quota_total=10)
    pcb.transition_to(AgentState.READY)
    pcb.transition_to(AgentState.RUNNING)
    pcb.transition_to(AgentState.BLOCKED)
    pcb.transition_to(AgentState.READY)  # resource granted -- re-arbitrate for a turn
    pcb.transition_to(AgentState.RUNNING)
    pcb.transition_to(AgentState.TERMINATED)
    assert pcb.state is AgentState.TERMINATED


def test_illegal_transition_raises():
    pcb = AgentPCB.new(priority=5, quota_total=10)
    with pytest.raises(ValueError):
        pcb.transition_to(AgentState.RUNNING)  # must go through READY first


def test_terminal_states_are_final():
    pcb = AgentPCB.new(priority=5, quota_total=10)
    pcb.transition_to(AgentState.READY)
    pcb.transition_to(AgentState.RUNNING)
    pcb.transition_to(AgentState.KILLED)
    with pytest.raises(ValueError):
        pcb.transition_to(AgentState.READY)


def test_same_state_transition_is_noop():
    pcb = AgentPCB.new(priority=5, quota_total=10)
    pcb.transition_to(AgentState.NEW)
    assert pcb.state is AgentState.NEW
