"""Dynamic direction detection system for vehicle tracking.

This module provides real-time direction analysis based on vehicle movement history,
replacing the static lane-based direction assignment system.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class DynamicDirectionAnalyzer:
    """Real-time direction detection based on vehicle movement history.
    
    This analyzer processes vehicle position history to determine movement direction
    dynamically, providing more accurate results than static lane assignments.
    """
    
    # Analysis parameters
    HISTORY_WINDOW_SECONDS = 3.0
    MIN_MOVEMENT_THRESHOLD_PIXELS = 15
    CONFIDENCE_THRESHOLD = 0.6
    
    # Direction stability parameters
    MIN_POSITIONS_FOR_ANALYSIS = 3
    HORIZONTAL_DOMINANCE_RATIO = 1.5  # dx must be 1.5x larger than dy for horizontal movement
    
    def analyze_movement_direction(
        self, 
        position_history: List[Tuple[float, Tuple[int, int]]]
    ) -> Tuple[Optional[str], float]:
        """Analyze vehicle movement to determine direction.
        
        Args:
            position_history: List of (timestamp, (x, y)) tuples
            
        Returns:
            (direction, confidence) where direction is "left"/"right"/"stationary"
            and confidence is a value between 0.0 and 1.0
        """
        if len(position_history) < self.MIN_POSITIONS_FOR_ANALYSIS:
            return None, 0.0
        
        # Filter recent positions within time window
        current_time = position_history[-1][0]
        recent_positions = [
            pos for timestamp, pos in position_history
            if current_time - timestamp <= self.HISTORY_WINDOW_SECONDS
        ]
        
        if len(recent_positions) < self.MIN_POSITIONS_FOR_ANALYSIS:
            return "stationary", 0.5
        
        # Calculate overall movement vector
        start_pos = recent_positions[0]
        end_pos = recent_positions[-1]
        
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        
        # Calculate movement distance
        movement_distance = (dx**2 + dy**2)**0.5
        
        if movement_distance < self.MIN_MOVEMENT_THRESHOLD_PIXELS:
            return "stationary", 0.8
        
        # Analyze movement pattern consistency
        consistency_score = self._calculate_movement_consistency(recent_positions)
        
        # Determine direction based on horizontal movement dominance
        direction, base_confidence = self._determine_direction(dx, dy, movement_distance)
        
        # Adjust confidence based on movement consistency
        final_confidence = min(0.95, base_confidence * consistency_score)
        
        logger.debug(
            f"Direction analysis: dx={dx}, dy={dy}, distance={movement_distance:.1f}, "
            f"consistency={consistency_score:.2f}, direction={direction}, confidence={final_confidence:.2f}"
        )
        
        return direction, final_confidence

    def _determine_direction(self, dx: float, dy: float, distance: float) -> Tuple[str, float]:
        """Determine movement direction from displacement vector.
        
        Args:
            dx: Horizontal displacement
            dy: Vertical displacement  
            distance: Total movement distance
            
        Returns:
            (direction, confidence) tuple
        """
        # Check if horizontal movement is dominant
        if abs(dx) > abs(dy) * self.HORIZONTAL_DOMINANCE_RATIO:
            # Clear horizontal movement
            direction = "right" if dx > 0 else "left"
            confidence = min(0.95, abs(dx) / distance)
        elif abs(dy) > abs(dx) * self.HORIZONTAL_DOMINANCE_RATIO:
            # Vertical movement - treat as stationary for traffic analysis
            direction = "stationary"
            confidence = 0.4
        else:
            # Diagonal or ambiguous movement
            if abs(dx) > abs(dy):
                # Slight horizontal preference
                direction = "right" if dx > 0 else "left"
                confidence = 0.5
            else:
                direction = "stationary"
                confidence = 0.3
        
        return direction, confidence

    def _calculate_movement_consistency(self, positions: List[Tuple[int, int]]) -> float:
        """Calculate consistency score for movement direction.
        
        A consistent movement has similar direction vectors between consecutive positions.
        
        Args:
            positions: List of (x, y) position tuples
            
        Returns:
            Consistency score between 0.0 and 1.0
        """
        if len(positions) < 3:
            return 0.5
        
        # Calculate direction vectors between consecutive positions
        direction_vectors = []
        for i in range(1, len(positions)):
            prev_x, prev_y = positions[i-1]
            curr_x, curr_y = positions[i]
            
            dx = curr_x - prev_x
            dy = curr_y - prev_y
            
            # Normalize vector (avoid division by zero)
            magnitude = (dx**2 + dy**2)**0.5
            if magnitude > 0:
                direction_vectors.append((dx/magnitude, dy/magnitude))
        
        if len(direction_vectors) < 2:
            return 0.5
        
        # Calculate average consistency between consecutive direction vectors
        consistency_scores = []
        for i in range(1, len(direction_vectors)):
            prev_dx, prev_dy = direction_vectors[i-1]
            curr_dx, curr_dy = direction_vectors[i]
            
            # Dot product gives cosine of angle between vectors
            dot_product = prev_dx * curr_dx + prev_dy * curr_dy
            # Convert to consistency score (1.0 = same direction, 0.0 = opposite)
            consistency = (dot_product + 1.0) / 2.0
            consistency_scores.append(consistency)
        
        # Return average consistency
        return sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.5

    def is_direction_reliable(self, direction: Optional[str], confidence: float) -> bool:
        """Check if direction detection is reliable enough to use.
        
        Args:
            direction: Detected direction
            confidence: Detection confidence
            
        Returns:
            True if direction is reliable enough to use
        """
        return (
            direction is not None 
            and confidence >= self.CONFIDENCE_THRESHOLD
        )

    def smooth_direction_transition(
        self, 
        current_direction: Optional[str], 
        new_direction: Optional[str], 
        new_confidence: float,
        current_confidence: float = 0.0
    ) -> Tuple[Optional[str], float]:
        """Smooth direction transitions to prevent flickering.
        
        Args:
            current_direction: Currently assigned direction
            new_direction: Newly detected direction
            new_confidence: Confidence in new direction
            current_confidence: Confidence in current direction
            
        Returns:
            (smoothed_direction, smoothed_confidence) tuple
        """
        # If no current direction, use new detection
        if current_direction is None:
            return new_direction, new_confidence
        
        # If new detection is unreliable, keep current
        if not self.is_direction_reliable(new_direction, new_confidence):
            return current_direction, max(0.1, current_confidence * 0.9)  # Decay confidence
        
        # If directions match, increase confidence
        if current_direction == new_direction:
            boosted_confidence = min(0.95, (current_confidence + new_confidence) / 2.0 + 0.1)
            return current_direction, boosted_confidence
        
        # If new direction is significantly more confident, switch
        confidence_diff = new_confidence - current_confidence
        if confidence_diff > 0.2:
            return new_direction, new_confidence
        
        # Otherwise, keep current direction but reduce confidence
        return current_direction, max(0.1, current_confidence * 0.8)


class MovementAnalytics:
    """Calculate movement analytics from position history."""
    
    @staticmethod
    def calculate_total_distance(position_history: List[Tuple[float, Tuple[int, int]]]) -> float:
        """Calculate total distance traveled in pixels.
        
        Args:
            position_history: List of (timestamp, (x, y)) tuples
            
        Returns:
            Total distance in pixels
        """
        if len(position_history) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(1, len(position_history)):
            _, prev_pos = position_history[i-1]
            _, curr_pos = position_history[i]
            
            dx = curr_pos[0] - prev_pos[0]
            dy = curr_pos[1] - prev_pos[1]
            
            distance = (dx**2 + dy**2)**0.5
            total_distance += distance
        
        return total_distance

    @staticmethod
    def calculate_average_speed(position_history: List[Tuple[float, Tuple[int, int]]]) -> float:
        """Calculate average speed in pixels per second.
        
        Args:
            position_history: List of (timestamp, (x, y)) tuples
            
        Returns:
            Average speed in pixels per second
        """
        if len(position_history) < 2:
            return 0.0
        
        total_distance = MovementAnalytics.calculate_total_distance(position_history)
        time_span = position_history[-1][0] - position_history[0][0]
        
        if time_span <= 0:
            return 0.0
        
        return total_distance / time_span

    @staticmethod
    def get_displacement_vector(position_history: List[Tuple[float, Tuple[int, int]]]) -> Tuple[float, float]:
        """Get overall displacement vector from start to end position.
        
        Args:
            position_history: List of (timestamp, (x, y)) tuples
            
        Returns:
            (dx, dy) displacement vector
        """
        if len(position_history) < 2:
            return 0.0, 0.0
        
        _, start_pos = position_history[0]
        _, end_pos = position_history[-1]
        
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        
        return dx, dy