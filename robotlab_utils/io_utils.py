"""
utility functions for file operations.
"""

import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator


def read_jsonl(file_path: Path) -> Generator[Dict[str, Any], None, None]:
    """
    Read JSONL file line by line.
    
    Args:
        file_path: Path to JSONL file
        
    Yields:
        Dictionary for each line
    """
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(data: List[Dict[str, Any]], file_path: Path):
    """
    Write list of dictionaries to JSONL file.
    
    Args:
        data: List of dictionaries
        file_path: Output file path
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w') as f:
        for record in data:
            f.write(json.dumps(record) + '\n')


def append_jsonl(record: Dict[str, Any], file_path: Path):
    """
    Append single record to JSONL file.
    
    Args:
        record: Dictionary to append
        file_path: Output file path
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'a') as f:
        f.write(json.dumps(record) + '\n')


def read_yolo_labels(label_path: Path) -> List[Dict[str, Any]]:
    """
    Read YOLO format label file.
    
    Args:
        label_path: Path to label file
        
    Returns:
        List of detection dictionaries
    """
    detections = []
    
    if not label_path.exists():
        return detections
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                det = {
                    'class_id': int(parts[0]),
                    'cx': float(parts[1]),
                    'cy': float(parts[2]),
                    'w': float(parts[3]),
                    'h': float(parts[4])
                }
                
                # Add confidence if present
                if len(parts) >= 6:
                    det['confidence'] = float(parts[5])
                
                detections.append(det)
    
    return detections


# def write_yolo_labels(detections: List[Dict[str, Any]],
#                      label_path: Path,
#                      include_conf: bool = True):
#     """
#     Write detections to YOLO format label file.
#
#     Args:
#         detections: List of detection dictionaries
#         label_path: Output file path
#         include_conf: Include confidence values
#     """
#     label_path.parent.mkdir(parents=True, exist_ok=True)
#
#     with open(label_path, 'w') as f:
#         for det in detections:
#             line = f"{det['class_id']} {det['cx']} {det['cy']} {det['w']} {det['h']}"
#
