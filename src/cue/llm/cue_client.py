import os
import logging

import httpx
from pydantic import BaseModel

from ..utils import DebugUtils
from ..config import get_settings
from ..schemas import AgentConfig, ErrorResponse, CompletionRequest, CompletionResponse
from .llm_request import LLMRequest
from .system_prompt import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class CueClient(LLMRequest):
    def __init__(
        self,
        config: AgentConfig,
    ):
        api_key = config.api_key or os.environ.get("CUE_API_KEY")
        if not api_key:
            raise ValueError("API key is missing in both config and settings.")

        self.api_key = api_key
        self.config = config
        self.model = config.model
        settings = get_settings()
        self.base_url = settings.get_base_url()
        logger.debug(f"[CueClient] initialized with model: {self.model} {self.config.id}")

    async def send_completion_request(self, request: CompletionRequest) -> CompletionResponse:
        response = None
        error = None

        try:
            messages = [
                msg.model_dump(exclude_none=True, exclude_unset=True) if isinstance(msg, BaseModel) else msg
                for msg in request.messages
            ]

            DebugUtils.debug_print_messages(messages, tag=f"{self.config.id} send_completion_request")

            # Prepare the system message and context
            if request.system_context:
                system_context = {"role": "assistant", "content": request.system_context.strip()}
                messages.insert(0, system_context)

            system_prompt = (
                f"{SYSTEM_PROMPT}{' ' + request.system_prompt_suffix if request.system_prompt_suffix else ''}"
            )
            system_message = {"role": "system", "content": system_prompt}
            messages.insert(0, system_message)
            request.messages = messages

            async with httpx.AsyncClient() as client:
                headers = {
                    "X-API-Key": f"{self.api_key}",
                    "Content-Type": "application/json",
                    "accept": "application/json",
                }

                response = await client.post(
                    f"{self.base_url}/chat/completions", json=request.model_dump(), headers=headers
                )

                if response.status_code != 200:
                    error = ErrorResponse(
                        message=f"API request failed with status {response.status_code}: {response.text}",
                        code=str(response.status_code),
                    )
                    return CompletionResponse(author=request.author, model=self.model, error=error)

                response_data = response.json()
                return CompletionResponse(**response_data)

        except httpx.RequestError as e:
            error = ErrorResponse(message=f"Request failed: {str(e)}")
        except Exception as e:
            error = ErrorResponse(message=f"Exception: {str(e)}")

        if error:
            logger.error(error.model_dump())
        return CompletionResponse(author=request.author, model=self.model, error=error)
