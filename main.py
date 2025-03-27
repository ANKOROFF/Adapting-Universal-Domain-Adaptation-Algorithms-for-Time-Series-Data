import argparse
import torch
import random
import numpy as np
from trainers.trainer import Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--da_method', type=str, default='DANCE')
    parser.add_argument('--source_dataset', type=str, default='WISDM')
    parser.add_argument('--target_dataset', type=str, default='WISDM')
    parser.add_argument('--source_domain', type=int, default=0)
    parser.add_argument('--target_domain', type=int, default=1)
    parser.add_argument('--backbone', type=str, default='CNN')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()
    
    # Set random seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Create trainer
    trainer = Trainer(
        source_dataset=args.source_dataset,
        target_dataset=args.target_dataset,
        source_domain=args.source_domain,
        target_domain=args.target_domain,
        backbone=args.backbone,
        da_method=args.da_method,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        save_dir=args.save_dir,
        device=args.device,
        verbose=args.verbose
    )
    
    # Train and evaluate
    trainer.train()
    
if __name__ == '__main__':
    main()
    