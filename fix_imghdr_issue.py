#!/usr/bin/env python3
"""
Script to fix the missing imghdr module issue in the YOLOv3 detector.
This script will modify the yolo_utils.py file to use PIL instead of imghdr.
"""

import os
import sys
import shutil

def fix_imghdr_issue():
    # Path to the yolo_utils.py file
    yolo_utils_path = os.path.join('hackathon', 'deep_sort_temp', 'detector', 'YOLOv3', 'yolo_utils.py')
    
    # Check if the file exists
    if not os.path.exists(yolo_utils_path):
        print(f"Error: Could not find {yolo_utils_path}")
        return False
    
    # Create a backup of the original file
    backup_path = yolo_utils_path + '.backup'
    shutil.copy2(yolo_utils_path, backup_path)
    print(f"Created backup at {backup_path}")
    
    # Read the file content
    with open(yolo_utils_path, 'r') as f:
        content = f.read()
    
    # Replace the imghdr import with PIL
    if 'import imghdr' in content:
        content = content.replace('import imghdr', 'from PIL import Image')
        
        # Replace any use of imghdr.what with PIL-based check
        if 'imghdr.what' in content:
            content = content.replace('imghdr.what', '# imghdr.what - replaced with PIL')
            
            # Find the get_image_size function and modify it
            # This is a simplified approach - you might need to adjust based on the actual code
            if 'def get_image_size(fname):' in content:
                old_func = 'def get_image_size(fname):'
                new_func = '''def get_image_size(fname):
    """
    Get image size using PIL instead of imghdr
    """
    try:
        with Image.open(fname) as img:
            return img.size[0], img.size[1]
    except:
        return -1, -1'''
                
                # Find the function and replace it
                start_idx = content.find('def get_image_size(fname):')
                if start_idx != -1:
                    # Find the end of the function
                    next_def = content.find('def ', start_idx + len('def get_image_size(fname):'))
                    if next_def != -1:
                        # Replace the entire function
                        content = content[:start_idx] + new_func + content[next_def:]
                    else:
                        # If there's no next function, just append the new one
                        content = content[:start_idx] + new_func
        
        # Write the modified content back to the file
        with open(yolo_utils_path, 'w') as f:
            f.write(content)
        
        print(f"Modified {yolo_utils_path} to use PIL instead of imghdr")
        return True
    else:
        print(f"No 'import imghdr' found in {yolo_utils_path}")
        return False

if __name__ == "__main__":
    print("Fixing imghdr module issue in YOLOv3 detector...")
    if fix_imghdr_issue():
        print("Fix applied successfully. Please try running your script again.")
    else:
        print("Could not apply fix automatically. You may need to modify the code manually.")
        print("Add 'from PIL import Image' and replace any imghdr.what calls with PIL-based alternatives.") 