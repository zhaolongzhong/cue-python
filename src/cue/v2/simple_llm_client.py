import os
from typing import Dict, List, Optional, AsyncGenerator
from dataclasses import dataclass

try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    from google import genai
except ImportError:
    genai = None


@dataclass
class SimpleMessage:
    role: str
    content: str


@dataclass
class SimpleResponse:
    content: str
    usage: Dict[str, int]
    model: str


class SimpleLLMClient:
    def __init__(self, model: str, api_key: Optional[str] = None):
        self.model = model.lower()
        self.provider = self._get_provider(model)
        self.client = self._create_client(api_key)

    def _get_provider(self, model: str) -> str:
        model_lower = model.lower()
        if "claude" in model_lower or "anthropic" in model_lower:
            return "anthropic"
        elif "gemini" in model_lower:
            return "gemini"
        elif "gpt" in model_lower or "openai" in model_lower:
            return "openai"
        else:
            return "openai"

    def _create_client(self, api_key: Optional[str] = None):
        if self.provider == "anthropic":
            if not anthropic:
                raise ImportError("anthropic library not installed")
            key = api_key or os.getenv("ANTHROPIC_API_KEY")
            return anthropic.AsyncAnthropic(api_key=key)
        elif self.provider == "gemini":
            if not genai:
                raise ImportError("google-generativeai library not installed")
            key = api_key or os.getenv("GEMINI_API_KEY")
            genai.configure(api_key=key)
            return genai.Client()
        else:  # openai
            if not openai:
                raise ImportError("openai library not installed")
            key = api_key or os.getenv("OPENAI_API_KEY")
            return openai.AsyncOpenAI(api_key=key)

    async def complete(self, messages: List[SimpleMessage]) -> SimpleResponse:
        if self.provider == "anthropic":
            return await self._anthropic_complete(messages)
        elif self.provider == "gemini":
            return await self._gemini_complete(messages)
        else:
            return await self._openai_complete(messages)

    async def stream_complete(self, messages: List[SimpleMessage]) -> AsyncGenerator[str, None]:
        if self.provider == "anthropic":
            async for chunk in self._anthropic_stream(messages):
                yield chunk
        elif self.provider == "gemini":
            async for chunk in self._gemini_stream(messages):
                yield chunk
        else:
            async for chunk in self._openai_stream(messages):
                yield chunk

    async def _anthropic_complete(self, messages: List[SimpleMessage]) -> SimpleResponse:
        system_msg = None
        formatted_msgs = []

        for msg in messages:
            if msg.role == "system":
                system_msg = msg.content
            else:
                formatted_msgs.append({"role": msg.role, "content": msg.content})

        kwargs = {"model": self.model, "max_tokens": 4096, "messages": formatted_msgs}
        if system_msg:
            kwargs["system"] = system_msg

        response = await self.client.messages.create(**kwargs)
        return SimpleResponse(
            content=response.content[0].text,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
            model=self.model,
        )

    async def _openai_complete(self, messages: List[SimpleMessage]) -> SimpleResponse:
        formatted_msgs = [{"role": msg.role, "content": msg.content} for msg in messages]

        response = await self.client.chat.completions.create(model=self.model, messages=formatted_msgs, max_tokens=4096)

        return SimpleResponse(
            content=response.choices[0].message.content,
            usage={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            model=self.model,
        )

    async def _gemini_complete(self, messages: List[SimpleMessage]) -> SimpleResponse:
        contents = []
        for msg in messages:
            if msg.role != "system":  # Gemini doesn't support system role in same way
                contents.append({"role": msg.role, "parts": [{"text": msg.content}]})

        response = await self.client.aio.models.generate_content(model=self.model, contents=contents)

        return SimpleResponse(
            content=response.text,
            usage={
                "input_tokens": 0,  # Gemini doesn't always provide usage
                "output_tokens": 0,
                "total_tokens": 0,
            },
            model=self.model,
        )

    async def _anthropic_stream(self, messages: List[SimpleMessage]) -> AsyncGenerator[str, None]:
        system_msg = None
        formatted_msgs = []

        for msg in messages:
            if msg.role == "system":
                system_msg = msg.content
            else:
                formatted_msgs.append({"role": msg.role, "content": msg.content})

        kwargs = {"model": self.model, "max_tokens": 4096, "messages": formatted_msgs, "stream": True}
        if system_msg:
            kwargs["system"] = system_msg

        async with self.client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text

    async def _openai_stream(self, messages: List[SimpleMessage]) -> AsyncGenerator[str, None]:
        formatted_msgs = [{"role": msg.role, "content": msg.content} for msg in messages]

        stream = await self.client.chat.completions.create(
            model=self.model, messages=formatted_msgs, max_tokens=4096, stream=True
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def _gemini_stream(self, messages: List[SimpleMessage]) -> AsyncGenerator[str, None]:
        contents = []
        for msg in messages:
            if msg.role != "system":
                contents.append({"role": msg.role, "parts": [{"text": msg.content}]})

        stream = await self.client.aio.models.generate_content_stream(model=self.model, contents=contents)

        async for chunk in stream:
            if chunk.text:
                yield chunk.text
