"""Module ButterworthLPF : filtre passe-bas de Butterworth pour un signal 1D."""

from scipy.signal import butter, lfilter_zi, lfilter
from typing import List


class ButterworthLPF:
    """Implémentation d'un filtre passe-bas de Butterworth pour un signal 1D.

    Attributes:
        b (List[float]): Coefficients du numérateur du filtre.
        a (List[float]): Coefficients du dénominateur du filtre.
        zi (List[float]): Condition initiale pour l'état du filtre.
    """

    def __init__(self, cutoff: float, fs: float, order: int = 2) -> None:
        """Initialise le filtre Butterworth passe-bas.

        Args:
            cutoff (float): Fréquence de coupure du filtre (Hz).
            fs (float): Fréquence d'échantillonnage du signal (Hz).
            order (int, optional): Ordre du filtre. Défaut à 2.
        """
        self.b: List[float]
        self.a: List[float]
        self.b, self.a = butter(order, cutoff / (0.5 * fs), btype="low")
        self.zi: List[float] = lfilter_zi(self.b, self.a)

    def update(self, x: float) -> float:
        """Met à jour le filtre avec une nouvelle valeur et renvoie la valeur filtrée.

        Args:
            x (float): Nouvelle valeur d'entrée à filtrer.

        Returns:
            float: Valeur filtrée correspondant à l'entrée x.
        """
        y, self.zi = lfilter(self.b, self.a, [x], zi=self.zi)
        return y[0]
