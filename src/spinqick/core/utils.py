from qick import QickConfig


def check_nyquist(freq: float, gen: int, soccfg: QickConfig) -> int:
    """Returns the correct nqz setting for a given generator and frequency."""
    gen_fs = soccfg["gens"][gen]["fs"]
    if freq > gen_fs / 2:
        nqz = 2
    else:
        nqz = 1
    return nqz
