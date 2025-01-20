from abc import ABC, abstractmethod

from ..types import CompletionRequest


class LLMRequest(ABC):
    @abstractmethod
    async def send_completion_request(self, request: CompletionRequest):
        pass
