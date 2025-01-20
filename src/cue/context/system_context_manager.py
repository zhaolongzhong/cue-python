import json
import logging
from typing import Optional
from datetime import datetime

from ..utils.token_counter import TokenCounter
from ..services.service_manager import ServiceManager

logger = logging.getLogger(__name__)


class SystemContextManager:
    def __init__(self, metrics: dict, token_stats: dict):
        self.metrics = metrics
        self.token_stats = token_stats
        self.token_counter = TokenCounter()
        self.system_context_base: Optional[str] = None
        self.system_context = ""
        self.service_manager: Optional[ServiceManager] = None

    def set_service_manager(self, service_manager: ServiceManager):
        self.service_manager = service_manager

    async def update_base_context(self) -> None:
        self.system_context_base = self._get_time_context()
        system_context = await self.service_manager.assistants.get_system_context()
        logger.info(f"update base system context: {system_context}")
        if system_context:
            self.system_context_base += f"<system_learning>{system_context}</system_learning>"

    def _get_time_context(self) -> str:
        return f"""
        * Time Awareness:
        - Recent Time: {datetime.today().strftime("%A, %B %-d, %Y %I:%M %p %Z")}
        - Note: This timestamp is for general context and system self-awareness only
        - Limitations: May have delay due to context caching
        - For precise time: Use bash tool with 'date' command or other available tool
        """

    def build_system_context(
        self,
        project_context: Optional[str] = None,
        task_context: Optional[str] = None,
        memories: Optional[str] = None,
        summaries: Optional[str] = None,
    ) -> str:
        """Build the system context."""
        new_system_context = self.system_context_base

        # Process project context
        if project_context:
            project_value, _ = self.update_stats("project", project_context, "project")
            new_system_context += project_value if project_value else ""

        # Process task context
        if task_context:
            task_value, _ = self.update_stats("task", task_context, "task")
            new_system_context += task_value if task_value else ""

        # Process recent memories
        if memories:
            memories_value, _ = self.update_stats("memories", memories, "memories")
            new_system_context += memories_value if memories_value else ""

        # Process message summaries
        if summaries:
            summaries_value, _ = self.update_stats("summaries", summaries, "summaries")
            new_system_context += summaries_value if summaries_value else ""

        # Update system context if changed
        if self.system_context != new_system_context:
            self.system_context = new_system_context
            self.token_stats["context_updated"] = True
            self.metrics["context_updated"] = True
            logger.debug(
                f"System context updated, \n{json.dumps({'old': self.system_context, 'new': new_system_context})}"
            )

        else:
            self.metrics["context_updated"] = False
            self.token_stats["context_updated"] = False

        return self.system_context

    def update_stats(self, key: str, current_value: str, context_key: str) -> tuple[str, int]:
        """
        Common method to update metrics and token stats for a given context component.
        """
        if not current_value:
            return "", 0

        # Update metrics stats
        local_stats = self.metrics.get(key, {"prev": "", "curr": ""})
        if local_stats.get("curr") != current_value:
            local_stats["prev"] = local_stats.get("curr", "")
            local_stats["curr"] = current_value
            local_stats["updated"] = True
        else:
            # it's same value in this turn
            local_stats["prev"] = local_stats.get("curr")
            local_stats["updated"] = False

        self.metrics[key] = local_stats

        # Count tokens
        tokens = self.token_counter.count_token(current_value)
        self.token_stats[context_key] = tokens

        # Log if needed
        logger.debug(f"{key}: {json.dumps(current_value, indent=4)}")

        return current_value, tokens
