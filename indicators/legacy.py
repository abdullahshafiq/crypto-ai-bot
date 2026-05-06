from .calc import calculate_base_indicators
from .signal import generate_quant_signal

def calculate_indicators(df):
    """Legacy alias for calculate_base_indicators."""
    return calculate_base_indicators(df)

def get_quant_signal(*args, **kwargs):
    """Legacy alias for generate_quant_signal."""
    return generate_quant_signal(*args, **kwargs)
