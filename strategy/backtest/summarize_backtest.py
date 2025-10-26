"""
Summarize Backtest Results
=========================

This script scans a backtest results folder, collects all metrics.json files,
and generates a comprehensive CSV summary.

Usage:
    from strategy.backtest.summarize_backtest import summarize_backtest

    # Generate summary CSV
    summarize_backtest("strategy/backtest/factor_report/cnn_04_09_72h_v1/20240701_20250331")

Output:
    Creates backtest_summary.csv in the specified folder with all metrics.
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any


def find_metrics_files(folder_path: str) -> List[str]:
    """
    Recursively find all metrics.json files in the given folder.

    Args:
        folder_path: Root folder path to search

    Returns:
        List of absolute paths to metrics.json files
    """
    metrics_files = []
    folder = Path(folder_path)

    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")

    # Recursively find all metrics.json files
    for metrics_file in folder.rglob("metrics.json"):
        metrics_files.append(str(metrics_file))

    return metrics_files


def extract_path_info(metrics_path: str, root_folder: str) -> Dict[str, str]:
    """
    Extract strategy type and config from the metrics file path.

    Args:
        metrics_path: Absolute path to metrics.json file
        root_folder: Root folder path

    Returns:
        Dictionary with strategy_type, config, and relative_path
    """
    metrics_path = Path(metrics_path)
    root_folder = Path(root_folder)

    # Get relative path from root folder
    relative_path = metrics_path.relative_to(root_folder)

    # Extract parent directories
    parts = relative_path.parts[:-1]  # Exclude 'metrics.json'

    if len(parts) >= 2:
        strategy_type = parts[-2]  # Second to last directory
        config = parts[-1]  # Last directory before metrics.json
    elif len(parts) == 1:
        strategy_type = "unknown"
        config = parts[0]
    else:
        strategy_type = "unknown"
        config = "unknown"

    return {
        "strategy_type": strategy_type,
        "config": config,
        "relative_path": str(relative_path.parent)
    }


def load_metrics(metrics_path: str) -> Dict[str, Any]:
    """
    Load metrics from a metrics.json file.

    Args:
        metrics_path: Path to metrics.json file

    Returns:
        Dictionary containing metrics data
    """
    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        return metrics
    except Exception as e:
        print(f"Error loading {metrics_path}: {e}")
        return {}


def summarize_backtest(folder_path: str) -> pd.DataFrame:
    """
    Summarize all backtest results in a folder into a CSV file.

    Args:
        folder_path: Path to the backtest results folder

    Returns:
        DataFrame containing all metrics
    """
    print(f"Scanning folder: {folder_path}")

    # Find all metrics.json files
    metrics_files = find_metrics_files(folder_path)
    print(f"Found {len(metrics_files)} metrics.json files")

    if len(metrics_files) == 0:
        print("No metrics.json files found!")
        return pd.DataFrame()

    # Collect all data
    all_data = []

    for metrics_path in metrics_files:
        # Extract path information
        path_info = extract_path_info(metrics_path, folder_path)

        # Load metrics
        metrics = load_metrics(metrics_path)

        if metrics:
            # Combine path info and metrics
            row_data = {**path_info, **metrics}
            all_data.append(row_data)

    # Create DataFrame
    df = pd.DataFrame(all_data)

    # Sort by strategy_type and config
    if not df.empty:
        df = df.sort_values(by=['strategy_type', 'config'])

    # Reorder columns: put identifying columns first
    if not df.empty:
        identifying_cols = ['strategy_type', 'config', 'relative_path']
        metric_cols = [col for col in df.columns if col not in identifying_cols]
        df = df[identifying_cols + metric_cols]

    # Save to CSV
    output_path = os.path.join(folder_path, "backtest_summary.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSummary saved to: {output_path}")
    print(f"Total rows: {len(df)}")

    # Print summary statistics
    if not df.empty and 'total_return' in df.columns:
        print("\n" + "="*60)
        print("Summary Statistics:")
        print("="*60)
        print(f"Average Total Return: {df['total_return'].mean():.4f}")
        print(f"Best Total Return: {df['total_return'].max():.4f}")
        print(f"Worst Total Return: {df['total_return'].min():.4f}")

        if 'sharpe_ratio' in df.columns:
            print(f"\nAverage Sharpe Ratio: {df['sharpe_ratio'].mean():.4f}")
            print(f"Best Sharpe Ratio: {df['sharpe_ratio'].max():.4f}")
            print(f"Worst Sharpe Ratio: {df['sharpe_ratio'].min():.4f}")

        if 'annual_return' in df.columns:
            print(f"\nAverage Annual Return: {df['annual_return'].mean():.4f}")
            print(f"Best Annual Return: {df['annual_return'].max():.4f}")
            print(f"Worst Annual Return: {df['annual_return'].min():.4f}")

        print("="*60)

    return df


if __name__ == "__main__":
    # Example usage - modify this path as needed
    folder_path = "./factor_report/cnn_04_09_72h_v1/20240701_20250331"

    # Check if running from project root
    if not os.path.exists(folder_path):
        # Try absolute path
        folder_path = "/home/craz/crypto/crypto-trading/strategy/backtest/factor_report/cnn_04_09_72h_v1/20240701_20250331"

    if os.path.exists(folder_path):
        df = summarize_backtest(folder_path)
        print(f"\nPreview of results:")
        print(df.head())
    else:
        print(f"Folder not found: {folder_path}")
        print("Please modify the folder_path in __main__ section or call summarize_backtest() with your folder path.")
