from typing import Union


SPM = 60 # seconds per minute
MPH = 60 # minutes per hour
def time_format(t: float) -> str:
    """
    Time format.

    Args:
        t (float): time (s)

    Returns:
        str: Formatted time string
    """
    h = t // (MPH * SPM)
    m = (t % (MPH * SPM)) // SPM
    s = t % SPM
    return f"{int(h):02}:{int(m):02}:{s:05.2f}"


Number = Union[int, float]
def value_format(value: Number, digits: int) -> str:
    """
    Value format.

    Args:
        value (Number): value
        digits (int): number of digits

    Returns:
        str: Formatted value string
    """
    num = round(value, digits)
    if digits > 0:
        text = f"{num:0.{digits}f}"
    else:
        text = str(int(num))
    return text