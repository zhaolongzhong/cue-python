import logging
from typing import Optional
from pathlib import Path

from ..utils.token_counter import TokenCounter
from ..services.service_manager import ServiceManager

logger = logging.getLogger(__name__)


class ProjectContextManager:
    def __init__(self, path: Optional[str], service_manager: Optional[ServiceManager] = None):
        self.path = path
        self.pre_context: Optional[str] = None
        self._project_context: Optional[str] = None
        self.token_counter = TokenCounter()
        self.message_params: Optional[dict] = None
        self.service_manager: Optional[ServiceManager] = service_manager

    async def update_context(self) -> None:
        """Load project context."""
        if self.service_manager:
            context = await self.service_manager.assistants.get_project_context()
            if context:
                self.pre_context = self._project_context
                self._project_context = str(context)
                self.update_params()
                return
            else:
                logger.debug("No project context")

        if not self.path:
            logger.debug("No project context path provided")
            return None
        try:
            context_path = Path(self.path)
            if context_path.exists():
                with open(context_path) as f:
                    self.pre_context = self._project_context
                    self._project_context = f.read()
                    self.update_params()

            else:
                logger.info(f"No project context provided, {context_path}")
        except Exception as e:
            logger.error(f"Failed to load project context: {e}")
        return None

    def update_params(self) -> Optional[dict]:
        tokens = self.token_counter.count_token(self._project_context)
        token_context = f"<project_context_token>{tokens}</project_context_token>"
        if not self.pre_context and not self._project_context:
            return None

        if self.pre_context and not self._project_context:
            self.message_params = {
                "role": "user",
                "content": (
                    f"Project context path: {self.path} <project_context></project_context>, "
                    "the content in the file has been overwritten with empty, if this is not expected "
                    "please revert or update, here is previous context: "
                    f"<pre_project_context>{self.pre_context}</pre_project_context> {token_context}"
                ),
            }
            self.pre_context = None
        else:
            content_prefix = f"Project context path: {self.path} " if not self.service_manager else ""
            self.message_params = {
                "role": "user",
                "content": f"{content_prefix}<project_context>\n{self._project_context}\n</project_context>\n"
                f"{token_context}",
            }
            self.pre_context = self.message_params

    def get_params(self) -> Optional[dict]:
        return self.message_params

    def get_project_context(self) -> Optional[str]:
        return self._project_context
