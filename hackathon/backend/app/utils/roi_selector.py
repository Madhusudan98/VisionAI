import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional

class ROISelector:
    """
    Utility for selecting regions of interest from a static frame.
    Allows users to mark polygons by clicking points on the image.
    """
    
    def __init__(self, window_name: str = "Region Selection"):
        self.window_name = window_name
        self.image = None
        self.original_image = None
        self.current_roi = []
        self.regions = {}
        self.current_region_name = ""
        self.is_drawing = False
        self.region_colors = {
            "cash_counter": (0, 255, 255),  # Yellow
            "entry_zone": (0, 255, 0),      # Green
            "exit_zone": (0, 0, 255),       # Red
            "cashier_zone": (255, 0, 0)     # Blue
        }
        
    def select_roi(self, image: np.ndarray, region_names: List[str]) -> Dict[str, np.ndarray]:
        """
        Main function to select regions of interest on the given image.
        
        Args:
            image: The image on which to select regions
            region_names: List of region names to select (in order)
            
        Returns:
            Dictionary of normalized region coordinates {region_name: np.array([[x1, y1], [x2, y2], ...])}
        """
        self.image = image.copy()
        self.original_image = image.copy()
        h, w = image.shape[:2]
        
        # Setup window and mouse callback
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        self.regions = {}
        
        # Instructions overlay
        self._draw_instructions(
            "Click to mark points. Press 'C' to close polygon, 'R' to reset, 'N' for next region, 'Q' to quit."
        )
        
        for region_name in region_names:
            self.current_region_name = region_name
            self.current_roi = []
            self.is_drawing = True
            
            print(f"Select points for {region_name} region. Press 'C' to close polygon when done.")
            self._draw_instructions(
                f"Selecting: {region_name} | Click to mark points. 'C' to close, 'R' to reset."
            )
            
            while self.is_drawing:
                cv2.imshow(self.window_name, self.image)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.current_roi = []
                    self.image = self.original_image.copy()
                    self._draw_existing_regions()
                    self._draw_instructions(
                        f"Selecting: {region_name} | Points reset. Click to mark new points."
                    )
                elif key == ord('c') and len(self.current_roi) >= 3:
                    # Close the polygon and save
                    self.regions[region_name] = np.array(self.current_roi)
                    self.is_drawing = False
                    self._draw_existing_regions()
                    self._draw_instructions(
                        f"{region_name} region saved. Press any key to continue."
                    )
                    cv2.imshow(self.window_name, self.image)
                    cv2.waitKey(0)
        
        cv2.destroyWindow(self.window_name)
        
        # Normalize coordinates
        normalized_regions = {}
        for name, points in self.regions.items():
            normalized_points = [[x/w, y/h] for x, y in points]
            normalized_regions[name] = np.array(normalized_points)
        
        return normalized_regions
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for region selection"""
        if not self.is_drawing:
            return
            
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_roi.append([x, y])
            # Draw the point
            cv2.circle(self.image, (x, y), 5, self.region_colors.get(self.current_region_name, (255, 255, 255)), -1)
            
            # Draw lines between points
            if len(self.current_roi) > 1:
                pt1 = tuple(self.current_roi[-2])
                pt2 = (x, y)
                cv2.line(self.image, pt1, pt2, self.region_colors.get(self.current_region_name, (255, 255, 255)), 2)
    
    def _draw_existing_regions(self):
        """Draw all existing regions on the image"""
        for name, points in self.regions.items():
            color = self.region_colors.get(name, (255, 255, 255))
            points_array = np.array(points, np.int32)
            cv2.polylines(self.image, [points_array], True, color, 2)
            
            # Add label
            if len(points) > 0:
                x, y = points[0]
                cv2.putText(self.image, name, (int(x), int(y) - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def _draw_instructions(self, text):
        """Draw instructions on the image"""
        # Create a semi-transparent overlay
        overlay = self.image.copy()
        h, w = self.image.shape[:2]
        cv2.rectangle(overlay, (0, h-40), (w, h), (0, 0, 0), -1)
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, self.image, 1 - alpha, 0, self.image)
        
        # Add text
        cv2.putText(self.image, text, (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2) 