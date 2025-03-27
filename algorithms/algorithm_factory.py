from algorithms.DANCE import DANCE
from algorithms.UAN import UAN
from algorithms.OVANet import OVANet
from algorithms.UniOT import UniOT
from algorithms.RAINCOAT import RAINCOAT

def get_algorithm_class(algorithm_type):
    """
    Returns the algorithm class based on the algorithm type
    Args:
        algorithm_type: str, type of algorithm (DANCE, UAN, OVANet, UniOT, RAINCOAT)
    Returns:
        algorithm class
    """
    algorithms = {
        "DANCE": DANCE,
        "UAN": UAN,
        "OVANet": OVANet,
        "UniOT": UniOT,
        "RAINCOAT": RAINCOAT,
        "Raincoat": RAINCOAT
    }
    
    if algorithm_type not in algorithms:
        raise ValueError(f"Algorithm {algorithm_type} not supported")
        
    return algorithms[algorithm_type] 