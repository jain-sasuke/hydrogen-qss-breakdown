import numpy as np
from scipy.interpolate import RegularGridInterpolator
from parser_adasf11 import parse_adf11

class ADASRateInterpolator:
    """
    Log-log interpolator for ADAS ADF11 rate coefficients.

    Interpolates:
        log10(K) over (log10(Te), log10(ne))

    Returns:
        K in cm^3/s (linear scale)
    """

    def __init__(self, filepath):

        Z, IDMAXD, ITMAXD, logne, logTe, logK = parse_adf11(filepath)

        # Hydrogen only → single block
        self.logTe = logTe
        self.logne = logne
        self.logK_grid = logK[0]  # shape (ITMAXD, IDMAXD)

        # Build interpolator in log space
        self.interpolator = RegularGridInterpolator(
            (self.logTe, self.logne),
            self.logK_grid,
            bounds_error=True,      # prevent extrapolation
            method="linear"
        )

    def __call__(self, Te_eV, ne_cm3):
        """
        Evaluate K(Te, ne).

        Inputs:
            Te_eV  : float
            ne_cm3 : float

        Returns:
            K [cm^3/s]
        """

        logTe = np.log10(Te_eV)
        logne = np.log10(ne_cm3)

        logK = self.interpolator((logTe, logne))

        return 10 ** logK
    
if __name__ == "__main__":

    scd = ADASRateInterpolator("data/raw/adas/scd96_h.dat")
    acd = ADASRateInterpolator("data/raw/adas/acd96_h.dat")

    Te_test = 5.0       # eV
    ne_test = 1e14      # cm^-3

    print("SCD(5eV, 1e14) =", scd(Te_test, ne_test))
    print("ACD(5eV, 1e14) =", acd(Te_test, ne_test))

    
    scd = ADASRateInterpolator("data/raw/adas/scd96_h.dat")

    ne_test = 1e14

    print("SCD at 2 eV =", scd(2.0, ne_test))
    print("SCD at 5 eV =", scd(5.0, ne_test))
    print("SCD at 10 eV =", scd(10.0, ne_test))