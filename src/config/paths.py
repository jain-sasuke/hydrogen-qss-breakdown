"""
Path Configuration for Non-Markovian CR Project

This module provides centralized path management to avoid hardcoded paths.
All paths are defined relative to the project root.

Usage:
    from src.config.paths import PATHS
    
    # Access paths
    ccc_raw_dir = PATHS['ccc_raw']
    output_dir = PATHS['ccc_processed']

Author: Week 2
Date: 2026-02-22
"""

from pathlib import Path
import os


def find_project_root():
    """
    Find project root by looking for pyproject.toml.
    
    Searches upward from current file location until finding
    the project root marker (pyproject.toml).
    
    Returns:
        Path: Absolute path to project root
    
    Raises:
        FileNotFoundError: If project root cannot be found
    """
    current = Path(__file__).resolve()
    
    # Search upward for pyproject.toml
    for parent in [current] + list(current.parents):
        if (parent / 'pyproject.toml').exists():
            return parent
    
    # Fallback: check for other markers
    for parent in [current] + list(current.parents):
        if (parent / 'README.md').exists() and (parent / 'data').exists():
            return parent
    
    raise FileNotFoundError(
        "Could not find project root. "
        "Make sure pyproject.toml exists in project root."
    )


# Auto-detect project root
PROJECT_ROOT = find_project_root()


# Define all paths relative to project root
PATHS = {
    # Project root
    'root': PROJECT_ROOT,
    
    # Data directories - Raw
    'data_raw': PROJECT_ROOT / 'data' / 'raw',
    'adas_raw': PROJECT_ROOT / 'data' / 'raw' / 'adas',
    'ccc_raw': PROJECT_ROOT / 'data' / 'raw' / 'ccc' / 'e-H_XSEC_LS',
    'hoang_binh_raw': PROJECT_ROOT / 'data' / 'raw' / 'hoang_binh',
    
    # Data directories - Processed
    'data_processed': PROJECT_ROOT / 'data' / 'processed',
    'adas_processed': PROJECT_ROOT / 'data' / 'processed' / 'adas',
    'ccc_processed': PROJECT_ROOT / 'data' / 'processed' / 'collisions' / 'ccc',
    'radiative_processed': PROJECT_ROOT / 'data' / 'processed' / 'Radiative',
    
    # Source code
    'src': PROJECT_ROOT / 'src',
    'parsers': PROJECT_ROOT / 'src' / 'parsers',
    'config': PROJECT_ROOT / 'src' / 'config',
    
    # Figures
    'figures': PROJECT_ROOT / 'figures',
    'figures_week1': PROJECT_ROOT / 'figures' / 'week1',
    'figures_week2': PROJECT_ROOT / 'figures' / 'week2',
    
    # Results
    'results': PROJECT_ROOT / 'results',
    
    # Reports
    'reports': PROJECT_ROOT / 'reports',
    
    # Notebooks
    'notebooks': PROJECT_ROOT / 'notebooks',
}


# Specific file paths (commonly used)
FILES = {
    # ADAS
    'adas_scd96': PATHS['adas_raw'] / 'scd96_h.dat',
    'adas_acd96': PATHS['adas_raw'] / 'acd96_h.dat',
    
    # Hoang-Binh
    'hoang_binh_csv': PATHS['radiative_processed'] / 'H_A_E1_LS_n1_15_physical.csv',
    
    # CCC Database
    'ccc_database_pkl': PATHS['ccc_processed'] / 'ccc_database.pkl',
    'ccc_summary_csv': PATHS['ccc_processed'] / 'ccc_transitions_summary.csv',
}


def ensure_dir(path_key):
    """
    Ensure a directory exists, create if necessary.
    
    Args:
        path_key: Key from PATHS dictionary
    
    Returns:
        Path: The directory path
    """
    path = PATHS[path_key]
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_path(key):
    """
    Get a path from the PATHS dictionary.
    
    Args:
        key: Path key (e.g., 'ccc_raw', 'adas_processed')
    
    Returns:
        Path: The requested path
    
    Raises:
        KeyError: If key not found
    """
    if key not in PATHS:
        available = ', '.join(PATHS.keys())
        raise KeyError(
            f"Path key '{key}' not found. "
            f"Available keys: {available}"
        )
    return PATHS[key]


def get_file(key):
    """
    Get a file path from the FILES dictionary.
    
    Args:
        key: File key (e.g., 'adas_scd96', 'ccc_database_pkl')
    
    Returns:
        Path: The requested file path
    
    Raises:
        KeyError: If key not found
    """
    if key not in FILES:
        available = ', '.join(FILES.keys())
        raise KeyError(
            f"File key '{key}' not found. "
            f"Available keys: {available}"
        )
    return FILES[key]


# Verification function (run on import to catch issues early)
def verify_paths():
    """
    Verify that critical paths exist.
    
    Prints warnings for missing paths but doesn't raise errors
    (some paths are created during execution).
    """
    critical_paths = ['data_raw', 'data_processed', 'src']
    
    for key in critical_paths:
        path = PATHS[key]
        if not path.exists():
            print(f"Warning: Path does not exist: {path}")
            print(f"  (Key: '{key}')")


# Auto-verify on import (optional, comment out if too verbose)
# verify_paths()


if __name__ == "__main__":
    print("=" * 70)
    print("PATH CONFIGURATION")
    print("=" * 70)
    print()
    print(f"Project root: {PROJECT_ROOT}")
    print()
    print("Key paths:")
    print("-" * 70)
    
    key_paths = [
        'ccc_raw',
        'ccc_processed',
        'adas_raw',
        'adas_processed',
        'radiative_processed',
    ]
    
    for key in key_paths:
        path = PATHS[key]
        exists = "EXISTS" if path.exists() else "MISSING"
        print(f"{key:25s} -> {exists:10s} | {path}")
    
    print()
    print("=" * 70)