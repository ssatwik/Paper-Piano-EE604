import cv2
from cv2.typing import MatLike
import numpy as np

class TouchValidator():
    def __init__(self, probbing_area: int = 6):
        self._probbing_area = probbing_area
        self._templates = []
        self._piano_points = []
        self._is_pressed = []
        
        self.test = []
    
    def initialize_points_images(self, frame: MatLike, piano_points: list[tuple[float, float]]):
        """Creates image for for every point to be used later as a template"""
        self._templates = []
        self._piano_points = piano_points
        self._is_pressed = [False] * len(piano_points)
        
        height, width = frame.shape[:2]
    
        for pp in piano_points:
            cropped = self._crop(frame, pp)
            self._templates.append(cropped.copy())
        
    def process(self, frame: MatLike, match_threshold: float = 0.7) -> set[int]:
        """Detects piano points hidden behind finger, return indexes of those points """
        result = set() # Set of indicies
        height, width = frame.shape[:2] 
        for indx, pp in enumerate(self._piano_points):
            cropped = self._crop(frame, pp)
            template = self._templates[indx]
            if cropped.size == 0 or cropped.shape != template.shape:
                self._is_pressed[indx] = False
                continue
            
            # Tries to match a point to corresponding cropped frame
            # If point is not detected it means there in no finger over it
            is_point_detected, image = self._is_template_in_image(cropped, template, match_threshold)
            
            if is_point_detected:
                if not self._is_pressed[indx]:
                    # Point was not detected and wasnt previously pressed
                    result.add(indx)
                    self._is_pressed[indx] = True
            else:
                # Point was detected
                self._is_pressed[indx] = False            
        
        return result    

    def scores(self, frame: MatLike) -> list[float]:
        """Return same-size patch NCC values without mutating press state."""
        values = []
        for point, template in zip(self._piano_points, self._templates):
            cropped = self._crop(frame, point)
            if cropped.size == 0 or cropped.shape != template.shape:
                values.append(-1.0)
                continue
            result = cv2.matchTemplate(cropped, template, cv2.TM_CCOEFF_NORMED)
            _, maximum, _, _ = cv2.minMaxLoc(np.nan_to_num(result, nan=-1.0))
            values.append(float(maximum))
        return values

    def _crop(self, frame, pp):
        height, width = frame.shape[:2]
        x = int(max(0, pp[0] - self._probbing_area))
        y = int(max(0, pp[1] - self._probbing_area + 1))
        right = int(min(x + self._probbing_area * 2, width))
        bottom = int(min(y + (self._probbing_area - 1) * 2, height))
        return frame[y:bottom, x:right]
    
    def _is_template_in_image(self, image, template, threshold=0.7):
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        
        # Get the maximum match value
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        # If the maximum match value is above the threshold, the object is present
        if max_val >= threshold:
            return False, result  # Point is present
    
        return True, result  # No Point detected
