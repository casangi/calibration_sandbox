import numpy as np

from abc import ABC, abstractmethod
from toolviper.utils import logger

class JonesMatrix(ABC):

    def __init__(self):
        pass

    @property
    @abstractmethod
    def shape(self)->tuple:
        raise NotImplementedError

    @property
    @abstractmethod
    def type(self)->str:
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self)->str:
        raise NotImplementedError

    @property
    @abstractmethod
    def dtype(self)->np.dtype:
        raise NotImplementedError

    @property
    @abstractmethod
    def matrix_type(self)->str:
        raise NotImplementedError

    @property
    @abstractmethod
    def polarization_basis(self)->str:
        raise NotImplementedError

    @property
    @abstractmethod
    def n_polarization(self)->int:
        raise NotImplementedError

    @property
    @abstractmethod
    def n_time(self)->int:
        raise NotImplementedError

    @property
    @abstractmethod
    def n_parameters(self)->int:
        raise NotImplementedError

    @property
    @abstractmethod
    def n_channel_parameters(self)->int:
        raise NotImplementedError

    @property
    @abstractmethod
    def n_channel_matrices(self)->int:
        raise NotImplementedError

    @property
    @abstractmethod
    def parameters(self)->np.ndarray:
        raise NotImplementedError

    @property
    @abstractmethod
    def matrix(self)->np.ndarray:
        raise NotImplementedError

    @property
    @abstractmethod
    def file_name(self)->str:
        raise NotImplementedError
