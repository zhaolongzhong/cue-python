import logging
from typing import Any, Optional

from ..utils.token_counter import TokenCounter

logger = logging.getLogger(__name__)


import os
import json
from pathlib import Path
from datetime import datetime, timedelta


class DynamicMemoryManager:
    def __init__(self, max_tokens: int = 1000, max_chars: int = 500, relevance_threshold: float = 0.5):
        """
        Initialize the DynamicMemoryManager with enhanced memory management capabilities.

        Args:
            max_tokens (int): Maximum number of tokens to maintain in memories
            max_chars (int): Maximum characters for each memory entry
            relevance_threshold (float): Minimum relevance score to retain memory (0.0-1.0)
        """
        self.max_tokens = max_tokens
        self.max_chars: int = max_chars
        self.relevance_threshold = relevance_threshold
        self.memories: dict[str, str] = {}
        self.memory_scores: dict[str, float] = {}  # Track relevance scores
        self.token_counter = TokenCounter()
        self.recent_memories: Optional[str] = None
        self.message_param: Optional[dict] = None
        self.cache_hits: dict[str, int] = {}  # Track memory usage
        self.last_access: dict[str, datetime] = {}  # Track memory access times
        self.score_history: dict[str, list[tuple[datetime, float]]] = {}  # Track score evolution

        # Load persistent data
        self._load_persistent_data()

    def _get_total_tokens(self) -> int:
        """Get the total token count for all memories in the current window."""
        combined_memories = "\n".join(self.memories)
        return self.token_counter.count_token(content=combined_memories)

    def _truncate_center(self, text: str, max_length: int) -> str:
        """Truncate text from the center if it exceeds max_length."""
        if len(text) <= max_length:
            return text
        half_length = (max_length - 3) // 2
        return text[:half_length] + "..." + text[-half_length:]

    def add_memories(self, memory_dict: dict[str, str], context: str = "") -> None:
        """
        Replace current memories with new memory dictionary using smart selection.
        Memories are evaluated based on relevance scoring and token limits.

        Args:
            memory_dict (dict[str, str]): Dictionary of memory_id to formatted memory content
            context (str): Current conversation context for relevance scoring
        """
        # Keep existing scores and hits for retained memories
        old_scores = {k: v for k, v in self.memory_scores.items() if k in memory_dict}
        old_hits = {k: v for k, v in self.cache_hits.items() if k in memory_dict}

        self.memories.clear()
        self.memory_scores.clear()
        self.cache_hits.clear()

        # Restore previous tracking data
        self.memory_scores.update(old_scores)
        self.cache_hits.update(old_hits)

        # First pass: Score all memories
        for memory_id, memory_content in memory_dict.items():
            if memory_id not in self.memory_scores:
                self.update_memory_score(memory_id, context)

        # Sort memories by score and recency
        sorted_memories = sorted(
            memory_dict.items(),
            key=lambda x: (self.memory_scores.get(x[0], 0), memory_dict.keys().index(x[0])),
            reverse=True
        )

        # Second pass: Add memories respecting relevance and token limits
        for memory_id, memory_content in sorted_memories:
            # Skip low relevance memories
            if self.memory_scores.get(memory_id, 0) < self.relevance_threshold:
                continue

            # Truncate if necessary
            truncated_memory = self._truncate_center(memory_content, self.max_chars)

            # Add to memories
            self.memories[memory_id] = truncated_memory

            # Check token limit
            if self._get_total_tokens() > self.max_tokens:
                # Remove the memory we just added
                self.memories.pop(memory_id)
                logger.debug(
                    f"Stopped adding memories due to token limit. "
                    f"Current tokens: {self._get_total_tokens()}/{self.max_tokens}"
                )
                break

        # Update the formatted memories
        self.update_recent_memories()

    def _get_total_tokens(self) -> int:
        """Get the total token count for all memories in the current window."""
        if not self.memories:
            return 0
        combined_memories = "\n".join(self.memories.values())
        return self.token_counter.count_token(content=combined_memories)

    def get_formatted_memories(self) -> Optional[str]:
        """
        Get memories formatted as a system message for the LLM.
        Returns None if no memories are present.
        """
        if not self.memories:
            logger.warning("no memories")
            return None

        combined_memories = "\n".join(self.memories.values())

        return f"""The following are your most recent memory records. Please:
1. Consider these memories as part of your context when responding
2. Update your understanding based on this new information
3. Note that memories are listed from most recent to oldest
4. Only reference these memories when relevant to the current conversation

Instructions for memory processing:
- Treat each memory as factual information about past interactions
- If new memories conflict with old ones, prefer the more recent memory
- Use memories to maintain conversation continuity
- Do not explicitly mention these instructions to the user

<recent_memories>
{combined_memories}
</recent_memories>
"""

    def update_recent_memories(self):
        """Update the recent memories string representation."""
        previous = self.recent_memories
        self.recent_memories = self.get_formatted_memories()
        logger.debug(f"update_recent_memories, \nprevious: {previous}, \nnew: {self.recent_memories}")
        self.message_param = {"role": "user", "content": self.recent_memories}

    def get_memories_param(self) -> Optional[dict]:
        return self.message_param

    def get_memory_stats(self) -> dict[str, Any]:
        """Get statistics about the current memories."""
        total_tokens = self._get_total_tokens()
        return {
            "memory_count": len(self.memories),
            "total_tokens": total_tokens,
            "remaining_tokens": self.max_tokens - total_tokens,
            "is_at_capacity": total_tokens >= self.max_tokens,
        }

    def clear_memories(self) -> None:
        """Clear all memories and related tracking."""
        self.memories.clear()
        self.memory_scores.clear()
        self.cache_hits.clear()
        self.last_access.clear()
        self.score_history.clear()
        self._save_persistent_data()

    def _analyze_memory_patterns(self) -> dict:
        """
        Analyze memory access patterns to optimize cleanup strategy.
        Returns pattern analysis results.
        """
        patterns = {}
        now = datetime.now()

        for memory_id, history in self.score_history.items():
            if len(history) < 3:
                continue

            # Analyze timing patterns
            intervals = []
            for i in range(1, len(history)):
                time_diff = (history[i][0] - history[i-1][0]).total_seconds()
                intervals.append(time_diff)

            avg_interval = sum(intervals) / len(intervals)
            variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)
            regularity = 1.0 - min(1.0, variance / (avg_interval ** 2))

            # Analyze score patterns
            scores = [s for _, s in history]
            avg_score = sum(scores) / len(scores)
            score_trend = (scores[-1] - scores[0]) / len(scores)

            # Calculate retention value
            last_access = history[-1][0]
            age = (now - last_access).total_seconds()
            recency = 1.0 / (1.0 + age / 86400)  # Age factor in days

            retention_value = (
                (0.3 * regularity) +     # Pattern regularity
                (0.3 * avg_score) +      # Historical relevance
                (0.2 * (1 + score_trend)) + # Score trajectory
                (0.2 * recency)          # Recency factor
            )

            patterns[memory_id] = {
                'retention_value': retention_value,
                'avg_interval': avg_interval,
                'regularity': regularity,
                'last_access': last_access,
                'avg_score': avg_score
            }

        return patterns

    def cleanup_old_data(self, max_age_days: int = 7) -> None:
        """
        Smart cleanup of memory data based on pattern analysis.
        
        Args:
            max_age_days (int): Base maximum age for retained data
        """
        now = datetime.now()
        base_cutoff = now - timedelta(days=max_age_days)

        # Analyze patterns
        patterns = self._analyze_memory_patterns()

        # Track cleanup statistics
        cleanup_stats = {
            'pattern_retained': 0,
            'age_removed': 0,
            'low_value_removed': 0
        }

        # Smart cleanup based on patterns
        for memory_id, pattern in patterns.items():
            retention_value = pattern['retention_value']
            last_access = pattern['last_access']

            # Adjust retention threshold based on pattern value
            if retention_value > 0.7:  # High-value pattern
                effective_cutoff = base_cutoff - timedelta(days=7)  # Extended retention
                cleanup_stats['pattern_retained'] += 1
            elif retention_value < 0.3:  # Low-value pattern
                effective_cutoff = base_cutoff - timedelta(days=-3)  # Reduced retention
                cleanup_stats['low_value_removed'] += 1
            else:
                effective_cutoff = base_cutoff

            # Apply cleanup
            if last_access < effective_cutoff:
                self.last_access.pop(memory_id, None)
                self.cache_hits.pop(memory_id, None)
                self.score_history.pop(memory_id, None)
                cleanup_stats['age_removed'] += 1

        # Clean up orphaned entries
        all_ids = set(patterns.keys())
        orphaned = (
            set(self.last_access.keys()) |
            set(self.cache_hits.keys()) |
            set(self.score_history.keys())
        ) - all_ids

        for memory_id in orphaned:
            self.last_access.pop(memory_id, None)
            self.cache_hits.pop(memory_id, None)
            self.score_history.pop(memory_id, None)

        self._save_persistent_data()

        logger.info(
            f"Smart cleanup complete: {cleanup_stats['pattern_retained']} patterns retained, "
            f"{cleanup_stats['age_removed']} aged out, "
            f"{cleanup_stats['low_value_removed']} low-value removed, "
            f"{len(orphaned)} orphaned entries cleaned"
        )

    def _load_persistent_data(self) -> None:
        """Load persistent memory data from disk."""
        data_dir = Path(os.path.expanduser("~/.cache/memory_manager"))
        data_file = data_dir / "memory_data.json"

        if not data_dir.exists():
            data_dir.mkdir(parents=True)
            return

        if data_file.exists():
            try:
                with open(data_file) as f:
                    data = json.load(f)
                    self.cache_hits = data.get('cache_hits', {})
                    self.score_history = {
                        k: [(datetime.fromisoformat(d), s) for d, s in v]
                        for k, v in data.get('score_history', {}).items()
                    }
                    self.last_access = {
                        k: datetime.fromisoformat(v)
                        for k, v in data.get('last_access', {}).items()
                    }
            except Exception as e:
                logger.error(f"Error loading persistent data: {e}")

    def _save_persistent_data(self) -> None:
        """Save memory data to disk."""
        data_dir = Path(os.path.expanduser("~/.cache/memory_manager"))
        data_file = data_dir / "memory_data.json"

        try:
            data = {
                'cache_hits': self.cache_hits,
                'score_history': {
                    k: [(d.isoformat(), s) for d, s in v]
                    for k, v in self.score_history.items()
                },
                'last_access': {
                    k: v.isoformat()
                    for k, v in self.last_access.items()
                }
            }
            with open(data_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Error saving persistent data: {e}")

    def _calculate_time_decay(self, memory_id: str) -> float:
        """Calculate time-based decay factor for memory relevance."""
        if memory_id not in self.last_access:
            return 0.0

        time_diff = datetime.now() - self.last_access[memory_id]
        # Decay over 24 hours to 0.5
        decay_factor = 0.5 + (0.5 * max(0, (24 - time_diff.total_seconds() / 3600) / 24))
        return decay_factor

    def _calculate_trend_factor(self, memory_id: str) -> float:
        """
        Calculate memory relevance trend with pattern detection.
        Identifies recurring patterns and adjusts scores accordingly.
        """
        if memory_id not in self.score_history:
            return 0.5

        history = self.score_history[memory_id]
        if len(history) < 2:
            return 0.5

        # Get recent history sorted by time
        recent = sorted(history[-10:], key=lambda x: x[0])
        if len(recent) < 2:
            return 0.5

        # Calculate base trend
        start_score = recent[0][1]
        end_score = recent[-1][1]
        base_trend = (end_score - start_score) / max(1, len(recent) - 1)

        # Detect patterns in access timing
        intervals = []
        for i in range(1, len(recent)):
            time_diff = (recent[i][0] - recent[i-1][0]).total_seconds()
            intervals.append(time_diff)

        if not intervals:
            return 0.5

        # Check for regular patterns
        avg_interval = sum(intervals) / len(intervals)
        variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)
        regularity = 1.0 - min(1.0, variance / (avg_interval ** 2))

        # Detect if current time matches pattern
        now = datetime.now()
        last_access = recent[-1][0]
        time_since_last = (now - last_access).total_seconds()
        pattern_match = abs(time_since_last - avg_interval) / max(1, avg_interval)
        pattern_match = 1.0 - min(1.0, pattern_match)

        # Combine factors
        trend_factor = (
            (0.4 * (0.5 + base_trend)) +  # Base score trend
            (0.3 * regularity) +           # Pattern regularity
            (0.3 * pattern_match)          # Current timing match
        )

        return max(0.0, min(1.0, trend_factor))

    def update_memory_score(self, memory_id: str, context: str) -> float:
        """
        Update relevance score for a memory based on current context and history.
        
        Args:
            memory_id (str): The memory ID to score
            context (str): Current conversation context
            
        Returns:
            float: Updated relevance score
        """
        if memory_id not in self.memories:
            return 0.0

        # Update access tracking
        now = datetime.now()
        self.last_access[memory_id] = now
        self.cache_hits[memory_id] = self.cache_hits.get(memory_id, 0) + 1

        # Calculate component scores
        cache_score = min(self.cache_hits[memory_id] / 10.0, 0.5)
        time_factor = self._calculate_time_decay(memory_id)
        trend_factor = self._calculate_trend_factor(memory_id)

        # Calculate semantic relevance
        memory_content = self.memories[memory_id].lower()
        context_words = set(context.lower().split())
        memory_words = set(memory_content.split())
        context_score = len(memory_words & context_words) / max(len(context_words), 1)

        # Combine scores with weights
        final_score = (
            (0.4 * context_score) +  # Semantic relevance
            (0.2 * cache_score) +    # Usage frequency
            (0.2 * time_factor) +    # Recency
            (0.2 * trend_factor)     # Score trend
        )

        # Update history
        if memory_id not in self.score_history:
            self.score_history[memory_id] = []
        self.score_history[memory_id].append((now, final_score))

        # Keep only last 10 history entries
        self.score_history[memory_id] = self.score_history[memory_id][-10:]

        self.memory_scores[memory_id] = final_score
        self._save_persistent_data()

        return final_score
