"""
Factory module for creating backbone neural network architectures.
Provides functionality to instantiate different types of backbone networks based on configuration.
"""

from models.backbones import CNN, TCN, TCNAttention

def get_backbone_class(backbone_type):
    """
    Factory function to retrieve the appropriate backbone network class.
    
    Args:
        backbone_type (str): Type of backbone architecture to create
            Supported values: "CNN", "TCN", "TCNAttention"
            
    Returns:
        class: The corresponding backbone network class
        
    Raises:
        ValueError: If the specified backbone type is not supported
    """
    if backbone_type == "CNN":
        return CNN
    elif backbone_type == "TCN":
        return TCN
    elif backbone_type == "TCNAttention":
        return TCNAttention
    else:
        raise ValueError(f"Backbone {backbone_type} not supported") 