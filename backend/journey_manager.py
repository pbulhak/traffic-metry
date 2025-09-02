"""Journey ID management system for unique vehicle tracking.

This module provides global unique journey identifiers to replace the recycled
ByteTrack track_ids, enabling better logging and debugging capabilities.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)


class JourneyIDManager:
    """Manages global unique journey identifiers for vehicle tracking.
    
    This class creates human-readable journey IDs that remain unique across
    the entire application lifecycle, solving the ByteTrack ID recycling problem.
    """

    def __init__(self, id_format: str = "JOURNEY_{:06d}", start_counter: int = 0):
        """Initialize journey ID manager.
        
        Args:
            id_format: Format string for journey IDs (default: JOURNEY_000001)
            start_counter: Starting counter value for ID continuation after restart
        """
        self.id_format = id_format
        self.track_id_to_journey_id: dict[int, str] = {}
        self.journey_id_to_track_id: dict[str, int] = {}
        self.journey_counter = start_counter  # Continue from where we left off
        self.total_journeys_created = 0

        logger.info(f"JourneyIDManager initialized with start_counter={start_counter}")
        if start_counter > 0:
            logger.info(f"Continuing journey ID sequence from database (next: JOURNEY_{start_counter + 1:06d})")

    def create_journey_id(self, track_id: int) -> str:
        """Create new unique journey ID for ByteTrack track_id.
        
        Args:
            track_id: ByteTrack track ID (can be recycled)
            
        Returns:
            Unique journey ID string
            
        Raises:
            ValueError: If track_id already has an assigned journey_id
        """
        if track_id in self.track_id_to_journey_id:
            existing_id = self.track_id_to_journey_id[track_id]
            logger.warning(f"Track {track_id} already has journey ID: {existing_id}")
            return existing_id

        # Generate new journey ID (increment counter first)
        self.journey_counter += 1
        self.total_journeys_created += 1

        journey_id = self.id_format.format(self.journey_counter)

        # Create bidirectional mapping
        self.track_id_to_journey_id[track_id] = journey_id
        self.journey_id_to_track_id[journey_id] = track_id

        logger.info(f"New journey created: Track {track_id} -> {journey_id}")
        return journey_id

    def get_journey_id(self, track_id: int) -> str | None:
        """Get existing journey ID for track_id.
        
        Args:
            track_id: ByteTrack track ID
            
        Returns:
            Journey ID if exists, None otherwise
        """
        return self.track_id_to_journey_id.get(track_id)

    def get_track_id(self, journey_id: str) -> int | None:
        """Get track ID for journey ID.
        
        Args:
            journey_id: Journey ID string
            
        Returns:
            Track ID if exists, None otherwise
        """
        return self.journey_id_to_track_id.get(journey_id)

    def release_journey_id(self, track_id: int) -> str | None:
        """Release journey ID when vehicle exits tracking.
        
        Args:
            track_id: ByteTrack track ID to release
            
        Returns:
            Released journey ID if existed, None otherwise
        """
        journey_id = self.track_id_to_journey_id.pop(track_id, None)

        if journey_id:
            self.journey_id_to_track_id.pop(journey_id, None)
            logger.info(f"Journey released: Track {track_id} -> {journey_id}")
            return journey_id

        logger.warning(f"Attempted to release non-existent track_id: {track_id}")
        return None

    def is_journey_active(self, track_id: int) -> bool:
        """Check if journey is currently active.
        
        Args:
            track_id: ByteTrack track ID
            
        Returns:
            True if journey is active, False otherwise
        """
        return track_id in self.track_id_to_journey_id

    def get_active_journeys_count(self) -> int:
        """Get count of currently active journeys.
        
        Returns:
            Number of active journeys
        """
        return len(self.track_id_to_journey_id)

    def get_active_journey_ids(self) -> list[str]:
        """Get list of all active journey IDs.
        
        Returns:
            List of active journey ID strings
        """
        return list(self.track_id_to_journey_id.values())

    def get_statistics(self) -> dict[str, int]:
        """Get journey management statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "active_journeys": self.get_active_journeys_count(),
            "total_journeys_created": self.total_journeys_created,
            "current_counter": self.journey_counter,
            "completed_journeys": self.total_journeys_created - self.get_active_journeys_count()
        }

    def reset(self) -> None:
        """Reset journey ID manager (for testing purposes).
        
        Warning: This will clear all active journeys!
        """
        logger.warning("Resetting JourneyIDManager - all active journeys will be lost!")

        self.track_id_to_journey_id.clear()
        self.journey_id_to_track_id.clear()
        self.journey_counter = 0
        # Note: total_journeys_created is NOT reset to preserve lifetime statistics

    def cleanup_stale_journeys(self, active_track_ids: set[int]) -> int:
        """Cleanup journeys for track_ids that are no longer active.
        
        This method should be called periodically to prevent memory leaks
        if vehicle exit events are missed.
        
        Args:
            active_track_ids: Set of currently active track IDs from tracker
            
        Returns:
            Number of stale journeys cleaned up
        """
        stale_track_ids = set(self.track_id_to_journey_id.keys()) - active_track_ids

        cleaned_count = 0
        for track_id in stale_track_ids:
            journey_id = self.release_journey_id(track_id)
            if journey_id:
                cleaned_count += 1
                logger.info(f"Cleaned up stale journey: {journey_id} (track {track_id})")

        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} stale journeys")

        return cleaned_count


class TimestampedJourneyIDManager(JourneyIDManager):
    """Journey ID manager with timestamp-based IDs for enhanced debugging.
    
    Creates journey IDs with embedded timestamps for easier log correlation.
    """

    def __init__(self) -> None:
        # Format: JOURNEY_20250901_143052_001
        super().__init__(id_format="JOURNEY_{timestamp}_{:03d}")
        self._session_counter = 0

    def create_journey_id(self, track_id: int) -> str:
        """Create timestamped journey ID.
        
        Args:
            track_id: ByteTrack track ID
            
        Returns:
            Timestamped journey ID
        """
        if track_id in self.track_id_to_journey_id:
            return self.track_id_to_journey_id[track_id]

        # Generate timestamp-based ID
        self._session_counter += 1
        self.total_journeys_created += 1

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        journey_id = f"JOURNEY_{timestamp}_{self._session_counter:03d}"

        # Create bidirectional mapping
        self.track_id_to_journey_id[track_id] = journey_id
        self.journey_id_to_track_id[journey_id] = track_id

        logger.info(f"New timestamped journey created: Track {track_id} -> {journey_id}")
        return journey_id


class UUIDJourneyIDManager(JourneyIDManager):
    """Journey ID manager using UUIDs for maximum uniqueness.
    
    Uses UUID4 for globally unique journey IDs, suitable for distributed systems.
    """

    def __init__(self) -> None:
        super().__init__(id_format="JOURNEY_{uuid}")

    def create_journey_id(self, track_id: int) -> str:
        """Create UUID-based journey ID.
        
        Args:
            track_id: ByteTrack track ID
            
        Returns:
            UUID-based journey ID
        """
        if track_id in self.track_id_to_journey_id:
            return self.track_id_to_journey_id[track_id]

        # Generate UUID-based ID
        self.total_journeys_created += 1

        unique_uuid = str(uuid.uuid4())[:8]  # Use first 8 chars for readability
        journey_id = f"JOURNEY_{unique_uuid.upper()}"

        # Create bidirectional mapping
        self.track_id_to_journey_id[track_id] = journey_id
        self.journey_id_to_track_id[journey_id] = track_id

        logger.info(f"New UUID journey created: Track {track_id} -> {journey_id}")
        return journey_id
