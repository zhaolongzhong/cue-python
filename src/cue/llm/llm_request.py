from abc import ABC, abstractmethod
from typing import AsyncIterator

from cue.types.completion_response import CompletionResponse

from ..types import CompletionRequest


class LLMRequest(ABC):
    @abstractmethod
    async def send_completion_request(self, request: CompletionRequest):
        pass

    @abstractmethod
    async def send_streaming_completion_request(self, request: CompletionRequest) -> AsyncIterator[CompletionResponse]:
        pass
