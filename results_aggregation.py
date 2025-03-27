"""
Script for running multiple experiments and aggregating results.
Runs experiments across different datasets, domains, and backbone architectures.
"""

import subprocess
import pandas as pd
import os
import time
import re
from datetime import datetime

# Define experiment configurations
METHODS = ['Uni_OT', 'OVANet']  # Только UniOT и OVANet
BACKBONES = ['CNN', 'TCN', 'TCNAttention']  # Все три бэкбона

# Define scenarios as in the paper
SCENARIOS = [
    # WISDM → HHAR scenarios (W→H)
    {'source_dataset': 'WISDM', 'target_dataset': 'HHAR_SA', 'source_domain': '4', 'target_domain': '0'},
    {'source_dataset': 'WISDM', 'target_dataset': 'HHAR_SA', 'source_domain': '5', 'target_domain': '1'},
    {'source_dataset': 'WISDM', 'target_dataset': 'HHAR_SA', 'source_domain': '6', 'target_domain': '2'},
    {'source_dataset': 'WISDM', 'target_dataset': 'HHAR_SA', 'source_domain': '7', 'target_domain': '3'},
    {'source_dataset': 'WISDM', 'target_dataset': 'HHAR_SA', 'source_domain': '17', 'target_domain': '4'},
    {'source_dataset': 'WISDM', 'target_dataset': 'HHAR_SA', 'source_domain': '18', 'target_domain': '5'},
    {'source_dataset': 'WISDM', 'target_dataset': 'HHAR_SA', 'source_domain': '19', 'target_domain': '6'},
    {'source_dataset': 'WISDM', 'target_dataset': 'HHAR_SA', 'source_domain': '20', 'target_domain': '7'},
    {'source_dataset': 'WISDM', 'target_dataset': 'HHAR_SA', 'source_domain': '23', 'target_domain': '8'},
    
    # HHAR → WISDM scenarios (H→W)
    {'source_dataset': 'HHAR_SA', 'target_dataset': 'WISDM', 'source_domain': '0', 'target_domain': '4'},
    {'source_dataset': 'HHAR_SA', 'target_dataset': 'WISDM', 'source_domain': '1', 'target_domain': '5'},
    {'source_dataset': 'HHAR_SA', 'target_dataset': 'WISDM', 'source_domain': '2', 'target_domain': '6'},
    {'source_dataset': 'HHAR_SA', 'target_dataset': 'WISDM', 'source_domain': '3', 'target_domain': '7'},
    {'source_dataset': 'HHAR_SA', 'target_dataset': 'WISDM', 'source_domain': '4', 'target_domain': '17'},
    {'source_dataset': 'HHAR_SA', 'target_dataset': 'WISDM', 'source_domain': '5', 'target_domain': '18'},
    {'source_dataset': 'HHAR_SA', 'target_dataset': 'WISDM', 'source_domain': '6', 'target_domain': '19'},
    {'source_dataset': 'HHAR_SA', 'target_dataset': 'WISDM', 'source_domain': '7', 'target_domain': '20'},
    {'source_dataset': 'HHAR_SA', 'target_dataset': 'WISDM', 'source_domain': '8', 'target_domain': '23'},
    
    # WISDM → WISDM scenarios (из таблицы)
    {'source_dataset': 'WISDM', 'target_dataset': 'WISDM', 'source_domain': '3', 'target_domain': '2'},
    {'source_dataset': 'WISDM', 'target_dataset': 'WISDM', 'source_domain': '4', 'target_domain': '7'},
    {'source_dataset': 'WISDM', 'target_dataset': 'WISDM', 'source_domain': '13', 'target_domain': '15'},
    {'source_dataset': 'WISDM', 'target_dataset': 'WISDM', 'source_domain': '14', 'target_domain': '19'},
    {'source_dataset': 'WISDM', 'target_dataset': 'WISDM', 'source_domain': '27', 'target_domain': '28'},
    {'source_dataset': 'WISDM', 'target_dataset': 'WISDM', 'source_domain': '1', 'target_domain': '0'},
    {'source_dataset': 'WISDM', 'target_dataset': 'WISDM', 'source_domain': '1', 'target_domain': '3'},
    {'source_dataset': 'WISDM', 'target_dataset': 'WISDM', 'source_domain': '10', 'target_domain': '11'},
    {'source_dataset': 'WISDM', 'target_dataset': 'WISDM', 'source_domain': '22', 'target_domain': '17'},
    {'source_dataset': 'WISDM', 'target_dataset': 'WISDM', 'source_domain': '27', 'target_domain': '15'},
]

def parse_metrics(output):
    """
    Parse experiment output to extract metrics.
    
    Args:
        output (str): Raw output from experiment
        
    Returns:
        dict: Dictionary containing extracted metrics
    """
    metrics = {
        'accuracy': 0.0,
        'f1_score': 0.0,
        'loss': 0.0,
        'h_score': 0.0
    }
    
    try:
        if output is None:
            return metrics
            
        # Try to find the best metrics (usually at the end of training)
        best_h_match = re.search(r'Лучший H-score:\s*([\d.]+)', output)
        if best_h_match:
            metrics['h_score'] = float(best_h_match.group(1))
            
        # Find the last occurrence of accuracy and F1 score for target domain
        acc_matches = re.findall(r'Целевой домен - Accuracy:\s*([\d.]+)', output)
        f1_matches = re.findall(r'Целевой домен - Accuracy:.*?F1:\s*([\d.]+)', output)
        loss_matches = re.findall(r'Средняя потеря:\s*([\d.]+)', output)
        
        if acc_matches:
            metrics['accuracy'] = float(acc_matches[-1])
        if f1_matches:
            metrics['f1_score'] = float(f1_matches[-1])
        if loss_matches:
            metrics['loss'] = float(loss_matches[-1])
            
        # If no H-score found in best metrics, try to find the last one
        if metrics['h_score'] == 0.0:
            h_matches = re.findall(r'H-score:\s*([\d.]+)', output)
            if h_matches:
                metrics['h_score'] = float(h_matches[-1])
                
    except Exception as e:
        print(f"Error parsing metrics: {e}")
        print("Raw output:")
        print(output)
        
    return metrics

def run_experiment(method, backbone, scenario):
    """
    Run a single experiment with given configuration.
    
    Args:
        method (str): Domain adaptation method
        backbone (str): Network backbone architecture
        scenario (dict): Scenario configuration
        
    Returns:
        dict: Results including accuracy, F1 score, loss and H-score
    """
    cmd = [
        'python', 'main.py',
        '--da_method', method,
        '--backbone', backbone,
        '--source_dataset', scenario['source_dataset'],
        '--target_dataset', scenario['target_dataset'],
        '--source_domain', scenario['source_domain'],
        '--target_domain', scenario['target_domain'],
        '--device', 'cuda',
        '--batch_size', '32',
        '--num_epochs', '50',
        '--verbose'
    ]
    
    try:
        # Run the experiment and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='cp1251')
        
        # Check if the experiment failed
        if result.returncode != 0:
            print(f"Experiment failed with error:\n{result.stderr}")
            return None
            
        # Parse metrics from output
        metrics = parse_metrics(result.stdout)
        
        # Print raw output for debugging
        print("\nRaw output:")
        print(result.stdout)
        print("\nExtracted metrics:")
        print(metrics)
        
        return metrics
        
    except Exception as e:
        print(f"Error running experiment: {e}")
        return None

def calculate_dataset_metrics(df, dataset_pair):
    """
    Calculate metrics for a specific dataset pair (e.g., W→H, H→W or W→W).
    
    Args:
        df (pd.DataFrame): DataFrame with results
        dataset_pair (str): Dataset pair identifier ('W→H', 'H→W', or 'W→W')
        
    Returns:
        pd.DataFrame: DataFrame with calculated metrics
    """
    if dataset_pair == 'W→H':
        mask = (df['Source_Dataset'] == 'WISDM') & (df['Target_Dataset'] == 'HHAR_SA')
    elif dataset_pair == 'H→W':
        mask = (df['Source_Dataset'] == 'HHAR_SA') & (df['Target_Dataset'] == 'WISDM')
    else:  # W→W
        mask = (df['Source_Dataset'] == 'WISDM') & (df['Target_Dataset'] == 'WISDM')
    
    dataset_results = df[mask].groupby('Method').agg({
        'H_Score': ['mean', 'std'],
        'Accuracy': ['mean', 'std'],
        'F1_Score': ['mean', 'std']
    }).round(3)
    
    return dataset_results

def main():
    """
    Main function to run all experiments and save results.
    """
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Initialize results DataFrame
    results = []
    
    # Get current timestamp for the results file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f'results/experiments_{timestamp}.csv'
    
    print(f"Results will be saved to: {csv_path}")
    
    # Run all experiments
    for method in METHODS:
        for backbone in BACKBONES:
            for scenario in SCENARIOS:
                print(f"\nRunning experiment:")
                print(f"Method: {method}")
                print(f"Backbone: {backbone}")
                print(f"Scenario: {scenario}")
                
                # Run experiment
                metrics = run_experiment(method, backbone, scenario)
                
                if metrics:
                    # Add experiment configuration and results to DataFrame
                    result_row = {
                        'Method': method,
                        'Backbone': backbone,
                        'Source_Dataset': scenario['source_dataset'],
                        'Target_Dataset': scenario['target_dataset'],
                        'Source_Domain': scenario['source_domain'],
                        'Target_Domain': scenario['target_domain'],
                        'Accuracy': metrics['accuracy'],
                        'F1_Score': metrics['f1_score'],
                        'Loss': metrics['loss'],
                        'H_Score': metrics['h_score']
                    }
                    results.append(result_row)
                    
                    # Create DataFrame with all results
                    df = pd.DataFrame(results)
                    
                    # Save detailed results
                    df.to_csv(csv_path, index=False)
                    print("\nResults saved to CSV")
                    print(f"Current results:\n{df.tail(1)}")
                    
                    # Calculate and save summary statistics
                    wh_results = calculate_dataset_metrics(df, 'W→H')
                    hw_results = calculate_dataset_metrics(df, 'H→W')
                    ww_results = calculate_dataset_metrics(df, 'W→W')
                    
                    # Calculate overall averages
                    overall_results = df.groupby('Method').agg({
                        'H_Score': ['mean', 'std'],
                        'Accuracy': ['mean', 'std'],
                        'F1_Score': ['mean', 'std']
                    }).round(3)
                    
                    # Save summary results to CSV
                    wh_results.to_csv(f'results/summary_WH_{timestamp}.csv')
                    hw_results.to_csv(f'results/summary_HW_{timestamp}.csv')
                    ww_results.to_csv(f'results/summary_WW_{timestamp}.csv')
                    overall_results.to_csv(f'results/summary_overall_{timestamp}.csv')
                    
                    print("\nUpdated summary tables:")
                    print("\nW→H Results:")
                    print(wh_results)
                    print("\nH→W Results:")
                    print(hw_results)
                    print("\nW→W Results:")
                    print(ww_results)
                    print("\nOverall Results:")
                    print(overall_results)
                
                # Add small delay between experiments
                time.sleep(2)

if __name__ == "__main__":
    main() 