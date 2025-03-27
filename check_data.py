import torch
import numpy as np

def check_dataset(file_path):
    print(f"\nChecking file: {file_path}")
    try:
        data = torch.load(file_path)
        print("Keys:", data.keys())
        print("Samples shape:", data['samples'].shape)
        
        # Convert labels to numpy array for analysis
        labels = data['labels']
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
            
        print("Labels:", np.unique(labels))
        print("Labels distribution:", np.bincount(labels))
        
        # Check for NaN values in the dataset
        if np.isnan(data['samples']).any():
            print("Warning: NaN values found in samples!")
        if np.isnan(labels).any():
            print("Warning: NaN values found in labels!")
            
        # Check for infinite values in the dataset
        if np.isinf(data['samples']).any():
            print("Warning: Inf values found in samples!")
        if np.isinf(labels).any():
            print("Warning: Inf values found in labels!")
            
        # Analyze value ranges in the dataset
        print("Samples value range:", np.min(data['samples']), "to", np.max(data['samples']))
        print("Labels value range:", np.min(labels), "to", np.max(labels))
            
    except Exception as e:
        print(f"Error loading file: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Check training dataset
    check_dataset('data/WISDM/train_0.pt')
    # Check test dataset
    check_dataset('data/WISDM/test_0.pt') 