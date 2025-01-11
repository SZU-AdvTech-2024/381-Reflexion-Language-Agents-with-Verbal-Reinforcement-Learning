class BaseAgent:

    def __init__(self, max_step: int = 10):
        self.max_step: int = max_step
        self.step_n: int = 0

    def step(self) -> None:
        pass
    def run(self) -> None:
        pass

    def reset(self) -> None:
        self.step_n = 0
        pass
