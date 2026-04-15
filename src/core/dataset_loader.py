"""
Test Dataset Loader

Loads and parses test datasets (E_in_plane and Multipoles_in_plane)
for model evaluation.
"""

import os
import re
from pathlib import Path
import numpy as np
from typing import Dict, Tuple, List, Optional
import logging

from .data_generator import get_mode_list

logger = logging.getLogger(__name__)

class TestDatasetLoader:
    """Loads test datasets from text files."""
    
    def __init__(self, features_dir: str, targets_dir: str):
        """
        Initialize the loader.
        
        Args:
            features_dir: Directory containing E_in_plane files (e.g., 1000.txt)
            targets_dir: Directory containing Multipoles_in_plane files (e.g., Results_1000.txt)
        """
        self.features_dir = Path(features_dir)
        self.targets_dir = Path(targets_dir)
        
        if not self.features_dir.exists():
            logger.warning(f"Features directory not found: {self.features_dir}")
        if not self.targets_dir.exists():
            logger.warning(f"Targets directory not found: {self.targets_dir}")
            
    def _parse_feature_file(self, filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parse a single E_in_plane file.
        
        Format: c_theta c_phi power |E_theta| phase_theta |E_phi| phase_phi
        Phase is in degrees.
        
        Returns:
            E_theta (complex array), E_phi (complex array)
        """
        data = np.loadtxt(filepath)
        
        # Extract columns
        # c_theta = data[:, 0]
        # c_phi = data[:, 1]
        # power = data[:, 2]
        abs_E_theta = data[:, 3]
        phase_theta_deg = data[:, 4]
        abs_E_phi = data[:, 5]
        phase_phi_deg = data[:, 6]
        
        # Convert phase to radians and create complex arrays
        E_theta = abs_E_theta * np.exp(1j * np.deg2rad(phase_theta_deg))
        E_phi = abs_E_phi * np.exp(1j * np.deg2rad(phase_phi_deg))
        
        return E_theta, E_phi
        
    def _parse_target_file(self, filepath: Path, maxorder: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parse a single Multipoles_in_plane file.
        
        Format: Type(E/M) l m Re(coeff) Im(coeff)
        
        Returns:
            a_e (complex array), a_m (complex array)
        """
        # Read lines
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        # Initialize arrays
        mode_list = get_mode_list(maxorder)
        n_modes = len(mode_list)
        a_e = np.zeros(n_modes, dtype=complex)
        a_m = np.zeros(n_modes, dtype=complex)
        
        # Create mapping from (l, m) to index
        mode_idx = {mode: i for i, mode in enumerate(mode_list)}
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
                
            type_str = parts[0]
            l = int(parts[1])
            m = int(parts[2])
            re_val = float(parts[3])
            im_val = float(parts[4])
            
            if l > maxorder:
                continue
                
            idx = mode_idx.get((l, m))
            if idx is not None:
                val = re_val + 1j * im_val
                if type_str == 'E':
                    a_e[idx] = val
                elif type_str == 'M':
                    a_m[idx] = val
                    
        return a_e, a_m
        
    def load_dataset(self, maxorder: int, limit: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load the dataset.
        
        Args:
            maxorder: Maximum multipole order to extract
            limit: Maximum number of samples to load (for quick testing)
            
        Returns:
            E_theta_batch, E_phi_batch, a_e_batch, a_m_batch
        """
        logger.info(f"Loading test dataset from {self.features_dir} and {self.targets_dir}")
        
        # Find matching files and sort for reproducible order
        feature_files = sorted(list(self.features_dir.glob("*.txt")))
        
        samples = []
        
        for feat_file in feature_files:
            # Extract sample ID (e.g., 1000 from 1000.txt)
            match = re.match(r'(\d+)\.txt', feat_file.name)
            if not match:
                continue
                
            sample_id = match.group(1)
            target_file = self.targets_dir / f"Results_{sample_id}.txt"
            
            if target_file.exists():
                samples.append((feat_file, target_file))
                
            if limit and len(samples) >= limit:
                logger.info(f"Reached limit of {limit} samples (reproducibly ordered)")
                break
                
        if not samples:
            raise ValueError("No matching feature and target files found.")
            
        logger.info(f"Found {len(samples)} valid samples.")
        
        # Load data
        E_theta_list = []
        E_phi_list = []
        a_e_list = []
        a_m_list = []
        
        for i, (feat_file, target_file) in enumerate(samples):
            if i % 100 == 0:
                logger.info(f"Processing sample {i}/{len(samples)}")
                
            try:
                E_theta, E_phi = self._parse_feature_file(feat_file)
                a_e, a_m = self._parse_target_file(target_file, maxorder)
                
                E_theta_list.append(E_theta)
                E_phi_list.append(E_phi)
                a_e_list.append(a_e)
                a_m_list.append(a_m)
            except Exception as e:
                logger.warning(f"Error processing sample {feat_file.name}: {e}")
                
        # Stack into batches
        E_theta_batch = np.stack(E_theta_list)
        E_phi_batch = np.stack(E_phi_list)
        a_e_batch = np.stack(a_e_list)
        a_m_batch = np.stack(a_m_list)
        
        logger.info(f"Loaded dataset: {len(E_theta_batch)} samples")
        return E_theta_batch, E_phi_batch, a_e_batch, a_m_batch
