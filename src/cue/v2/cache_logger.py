"""
Cache Usage Logging Utility

Centralized logging for cache usage statistics across different components.
Supports both individual request/turn logging and conversation-level logging.
"""

import os
import json
from typing import Any, Dict, List, Optional
from datetime import datetime


class CacheLogger:
    """Centralized cache usage logger for development and analysis"""

    def __init__(self, log_file: Optional[str] = None):
        self.cache_logging_enabled = os.getenv("CUE_LOG_CACHE_USAGE", "false").lower() in ("true", "1", "yes")
        self.cache_log_file = log_file or os.getenv("CUE_CACHE_LOG_FILE", "temp/cache_usage_dev.jsonl")
        self.log_individual_turns = os.getenv("CUE_LOG_INDIVIDUAL_TURNS", "true").lower() in ("true", "1", "yes")
        self.log_conversations = os.getenv("CUE_LOG_CONVERSATIONS", "true").lower() in ("true", "1", "yes")

    def _calculate_cache_stats(self, usage: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate cache statistics from usage data"""
        cache_created = usage.get("cache_creation_input_tokens", 0)
        cache_read = usage.get("cache_read_input_tokens", 0)
        regular_input = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        total_input = regular_input + cache_created + cache_read

        # Cache efficiency calculation
        total_cache_tokens = cache_created + cache_read
        cache_efficiency = (cache_read / total_cache_tokens * 100) if total_cache_tokens > 0 else 0.0

        return {
            "cache_created": cache_created,
            "cache_read": cache_read,
            "regular_input": regular_input,
            "output": output_tokens,
            "total_input": total_input,
            "cache_hit": cache_read > 0,
            "cache_efficiency": round(cache_efficiency, 2),
        }

    def _ensure_log_directory(self) -> None:
        """Ensure the log directory exists"""
        log_dir = os.path.dirname(self.cache_log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def _write_log_entry(self, log_entry: Dict[str, Any]) -> None:
        """Write a log entry to the file"""
        try:
            self._ensure_log_directory()
            with open(self.cache_log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception:
            # Silent failure - don't break the main functionality
            pass

    def log_turn_usage(self, usage: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> None:
        """Log individual turn/request cache usage statistics"""
        if not self.cache_logging_enabled or not self.log_individual_turns:
            return

        cache_stats = self._calculate_cache_stats(usage)

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "usage": usage,
            "cache_stats": cache_stats,
            "context": context or {},
        }

        self._write_log_entry(log_entry)

    def log_conversation_usage(
        self,
        usage: Dict[str, Any],
        agent: Any,  # SimpleAgent type, but avoiding import
        steps: List[Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log conversation-level cache usage statistics"""
        if not self.cache_logging_enabled or not self.log_conversations:
            return

        cache_stats = self._calculate_cache_stats(usage)

        # Build context for conversation
        context = {
            "type": "conversation",
            "model": getattr(agent, "model", "unknown"),
            "turns": metadata.get("turns", len(steps)) if metadata else len(steps),
            "max_turns_reached": metadata.get("max_turns_reached", False) if metadata else False,
            "system_prompt_length": len(agent.system_prompt)
            if hasattr(agent, "system_prompt") and agent.system_prompt
            else 0,
            "tools_available": len(agent.tools) if hasattr(agent, "tools") and agent.tools else 0,
        }

        # Add any additional metadata
        if metadata:
            context.update({k: v for k, v in metadata.items() if k not in context})

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "usage": usage,
            "cache_stats": cache_stats,
            "context": context,
        }

        self._write_log_entry(log_entry)


# Global logger instance for convenience
_default_logger = None


def get_cache_logger() -> CacheLogger:
    """Get the default cache logger instance"""
    global _default_logger
    if _default_logger is None:
        _default_logger = CacheLogger()
    return _default_logger


def log_turn_usage(usage: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> None:
    """Convenience function for logging turn usage"""
    get_cache_logger().log_turn_usage(usage, context)


def log_conversation_usage(
    usage: Dict[str, Any], agent: Any, steps: List[Any], metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Convenience function for logging conversation usage"""
    get_cache_logger().log_conversation_usage(usage, agent, steps, metadata)
