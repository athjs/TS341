"""Filtre passe-bas Butterworth pour suivi de centroïdes."""

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, lfilter, lfilter_zi
from typing import Tuple, cast


class ButterworthLPF:
    """Filtre passe-bas Butterworth unidimensionnel.

    Attributes:
        cutoff (float): Fréquence de coupure normalisée (0 < cutoff < 0.5)
        fs (float): Fréquence d'échantillonnage
        order (int): Ordre du filtre
        b (NDArray[np.float64]): Coefficients du numérateur du filtre
        a (NDArray[np.float64]): Coefficients du dénominateur du filtre
        zi (NDArray[np.float64]): Conditions initiales du filtre
    """

    def __init__(self, cutoff: float, fs: float = 1.0, order: int = 3) -> None:
        """Docstring for __init__"""
        self.cutoff = cutoff
        self.fs = fs
        self.order = order

        # Conception du filtre Butterworth
        res = butter(self.order, self.cutoff, btype="low", fs=self.fs)
        # Cast explicite pour rassurer Pyright
        b_arr, a_arr = cast(Tuple[np.ndarray, np.ndarray], res)
        self.b: NDArray[np.float64] = np.asarray(b_arr, dtype=np.float64)
        self.a: NDArray[np.float64] = np.asarray(a_arr, dtype=np.float64)

        # Initialisation du filtre
        zi_arr = lfilter_zi(self.b, self.a)
        self.zi: NDArray[np.float64] = np.asarray(zi_arr, dtype=np.float64)

    def update(self, x: float) -> float:
        """Met à jour le filtre avec une nouvelle valeur et renvoie la valeur filtrée.

        Args:
            x (float): Nouvelle valeur à filtrer

        Returns:
            float: Valeur filtrée
        """
        y_raw, zi_new_raw = lfilter(self.b, self.a, [x], zi=self.zi)
        # Conversion explicite en float64 pour Pyright
        y: NDArray[np.float64] = np.asarray(y_raw, dtype=np.float64)
        zi_new: NDArray[np.float64] = np.asarray(zi_new_raw, dtype=np.float64)
        self.zi = zi_new
        return float(y[0])
