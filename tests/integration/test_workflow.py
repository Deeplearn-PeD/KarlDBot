import pytest

from karldbot.orchestration.coordinator import WorkflowStateMachine, AgentCoordinator
from karldbot.models.state import WorkflowState


class TestWorkflowStateMachine:
    def test_initial_state(self):
        sm = WorkflowStateMachine()
        assert sm.current_state == WorkflowState.INIT
        assert sm.history == [WorkflowState.INIT]

    def test_valid_transition(self):
        sm = WorkflowStateMachine()
        assert sm.can_transition_to(WorkflowState.CODING)
        assert sm.transition(WorkflowState.CODING)
        assert sm.current_state == WorkflowState.CODING
        assert len(sm.history) == 2

    def test_invalid_transition(self):
        sm = WorkflowStateMachine()
        assert not sm.can_transition_to(WorkflowState.COMPLETED)
        assert not sm.transition(WorkflowState.COMPLETED)
        assert sm.current_state == WorkflowState.INIT

    def test_reset(self):
        sm = WorkflowStateMachine()
        sm.transition(WorkflowState.CODING)
        sm.transition(WorkflowState.REVIEWING)
        sm.reset()
        assert sm.current_state == WorkflowState.INIT
        assert sm.history == [WorkflowState.INIT]

    def test_full_workflow(self):
        sm = WorkflowStateMachine()

        sm.transition(WorkflowState.CODING)
        assert sm.current_state == WorkflowState.CODING

        sm.transition(WorkflowState.REVIEWING)
        assert sm.current_state == WorkflowState.REVIEWING

        sm.transition(WorkflowState.DEBUGGING)
        assert sm.current_state == WorkflowState.DEBUGGING

        sm.transition(WorkflowState.CODING)
        assert sm.current_state == WorkflowState.CODING
