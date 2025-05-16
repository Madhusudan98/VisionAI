#!/usr/bin/env python3
"""
Script to disable MMDet import in the detector's __init__.py file.
This is useful if you don't need MMDet functionality and don't want to install it.
"""

import os
import sys
import shutil

def disable_mmdet():
    # Path to the detector's __init__.py file
    init_path = os.path.join('hackathon', 'deep_sort_temp', 'detector', '__init__.py')
    
    # Check if the file exists
    if not os.path.exists(init_path):
        print(f"Error: Could not find {init_path}")
        return False
    
    # Create a backup of the original file
    backup_path = init_path + '.backup'
    shutil.copy2(init_path, backup_path)
    print(f"Created backup at {backup_path}")
    
    # Read the file content
    with open(init_path, 'r') as f:
        content = f.read()
    
    # Check if MMDet import is present
    if 'from .MMDet import MMDet' in content:
        # Replace the import with a try-except block
        new_content = content.replace(
            'from .MMDet import MMDet',
            '# Optional import - will be skipped if MMDet is not installed\n'
            'try:\n'
            '    from .MMDet import MMDet\n'
            'except ImportError:\n'
            '    print("MMDet not available. If you need this functionality, install mmdet and mmcv-full.")\n'
            '    MMDet = None'
        )
        
        # Write the modified content back to the file
        with open(init_path, 'w') as f:
            f.write(new_content)
        
        print(f"Modified {init_path} to make MMDet import optional")
        return True
    else:
        print(f"No 'from .MMDet import MMDet' found in {init_path}")
        return False

if __name__ == "__main__":
    print("Disabling MMDet import in detector's __init__.py file...")
    if disable_mmdet():
        print("Fix applied successfully. Please try running your script again.")
    else:
        print("Could not apply fix automatically. You may need to modify the code manually.")
        print("Wrap the MMDet import in a try-except block to make it optional.") 