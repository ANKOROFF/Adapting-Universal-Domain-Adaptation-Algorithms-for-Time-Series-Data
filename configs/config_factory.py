from configs.HHAR_SA import HHAR_SA_Configs
from configs.HAR import HAR_Configs
from configs.WISDM import WISDM_Configs

def get_dataset_class(dataset_name):
    """
    Factory function to retrieve the appropriate dataset configuration class.
    
    Args:
        dataset_name (str): Name of the dataset to configure (e.g., "HHAR_SA", "HAR", "WISDM")
        
    Returns:
        class: The corresponding dataset configuration class
        
    Raises:
        ValueError: If the specified dataset is not supported
    """
    if dataset_name == "HHAR_SA":
        return HHAR_SA_Configs
    elif dataset_name == "HAR":
        return HAR_Configs
    elif dataset_name == "WISDM":
        return WISDM_Configs
    else:
        raise ValueError(f"Dataset {dataset_name} not supported") 