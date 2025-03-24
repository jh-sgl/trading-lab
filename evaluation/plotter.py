from abc import abstractmethod
from pathlib import Path


class Plotter:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def draw_plot(self, save_path: Path):
        pass
