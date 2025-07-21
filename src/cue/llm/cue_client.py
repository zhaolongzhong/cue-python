import os
import logging
from typing import AsyncIterator

import httpx
from pydantic import BaseModel

from ..types import AgentConfig, ErrorResponse, CompletionRequest, CompletionResponse
from ..utils import DebugUtils
from ..config import get_settings
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
            if "claude" not in request.model:
                system_message = {"role": "system", "content": system_prompt}
                messages.insert(0, system_message)
            request.messages = messages

            async with httpx.AsyncClient(
                transport=httpx.AsyncHTTPTransport(
                    http1=True,  # Explicitly use HTTP/1.1
                    http2=False,
                )
            ) as client:
                headers = {
                    "X-API-Key": f"{self.api_key}",
                    "Content-Type": "application/json",
                    "accept": "application/json",
                }

                try:
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        json=request.model_dump(),
                        headers=headers,
                        timeout=60.0,
                    )

                    logger.debug(f"Response status: {response.status_code}")
                    logger.debug(f"Response headers: {response.headers}")

                    if response.status_code != 200:
                        error_detail = {
                            "status_code": response.status_code,
                            "response_text": response.text,
                            "request_url": str(response.url),
                            "request_headers": dict(response.request.headers),
                        }
                        logger.error(f"API request failed: {error_detail}")

                        error = ErrorResponse(
                            message=f"API request failed with status {response.status_code}",
                            code=str(response.status_code),
                            details=error_detail,
                        )
                        return CompletionResponse(author=request.author, model=self.model, error=error)

                    response_data = response.json()
                    return CompletionResponse.parse_response_data(response_data=response_data, model=self.model)

                except httpx.TimeoutException as e:
                    error = ErrorResponse(
                        message=f"Request timed out: {str(e)}",
                        code="TIMEOUT",
                        details={
                            "timeout_seconds": 60.0,
                        },
                    )
                except httpx.RequestError as e:
                    error = ErrorResponse(
                        message=f"Request failed: {str(e)}",
                        code="REQUEST_ERROR",
                        details={"error_type": type(e).__name__, "base_url": self.base_url, "request_details": str(e)},
                    )

        except Exception as e:
            import traceback

            error = ErrorResponse(
                message=f"Exception: {str(e)}",
                code="INTERNAL_ERROR",
                details={"error_type": type(e).__name__, "traceback": traceback.format_exc()},
            )

        if error:
            logger.error(f"Error details: {error.model_dump()}")
        return CompletionResponse(author=request.author, model=self.model, error=error)

    async def send_streaming_completion_request(self, request: CompletionRequest) -> AsyncIterator[CompletionResponse]:
        """Streaming not implemented for Cue - yields single response."""
        response = await self.send_completion_request(request)
        yield response
