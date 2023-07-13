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
    m = t // MPH
    s = t % MPH
    return f'{int(h):02}:{int(m):02}:{s:05.2f}' 
