import numpy as np

from calviper.base import JonesMatrix
from typing import TypeVar, Type, Union

T = TypeVar('T', bound='Parent')

class GainJones(JonesMatrix):
    def __init__(self):
        super(GainJones, self).__init__()

        # public parent variable
        self.type: Union[str, None] = "G"
        self.dtype = np.complex64
        self.n_polarizations: Union[int, None] = 2
        self.n_parameters: Union[int, None] = 1
        self.channel_dependent_parameters: bool = False

        self.name: str = "GainJonesMatrix"

    # This is just an example of how this would be done. There should certainly be checks and customization
    # but for now just set the values simply as the original code doesn't do anything more complicated for now.
    @property
    def parameters(self) -> np.ndarray:
        return self._parameters

    @parameters.setter
    def parameters(self, array: np.ndarray) -> None:
        self._parameters = array

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    @matrix.setter
    def matrix(self, array: np.ndarray) -> None:
        self._matrix = array

    def calculate(self) -> None:
        self.initialize_jones()

        self.matrix = np.identity(2, dtype=np.complex64)
        self.matrix = np.tile(self.matrix, [self.n_times, self.n_antennas, self.n_channel_matrices, 1, 1])