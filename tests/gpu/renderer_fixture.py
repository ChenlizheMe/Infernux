"""Count completed Game graph executions in standalone renderer fixtures."""
from time import monotonic


class GameFrames:
    """Do not count owner-loop iterations while the first GPU graph is preparing."""

    def __init__(self, engine, *, timeout=60.0):
        self.engine = engine
        self.deadline = monotonic() + timeout
        self.previous = None

    def advance(self):
        snapshot = self.engine.renderer_frame_snapshot
        if monotonic() > self.deadline:
            raise AssertionError(f"Game GPU fixture timed out: {snapshot}")
        execution = snapshot["game_render_graph_execution_count"]
        if not self.engine.game_presentation_ready or execution == self.previous:
            return False
        self.previous = execution
        return True
