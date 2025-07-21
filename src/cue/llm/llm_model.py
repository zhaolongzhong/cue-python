import os
from enum import Enum


class ChatModel(Enum):
    # Claude https://docs.anthropic.com/en/docs/about-claude/models#model-names
    CLAUDE_OPUS_4_20250514 = ("claude-opus-4-20250514", True, "anthropic")
    CLAUDE_SONNET_4_20250514 = ("claude-sonnet-4-20250514", True, "anthropic")
    CLAUDE_3_7_SONNET_20250219 = ("claude-3-7-sonnet-20250219", True, "anthropic")
    CLAUDE_3_OPUS_20240229 = ("claude-3-opus-20240229", True, "anthropic")
    CLAUDE_3_5_SONNET_20241022 = ("claude-3-5-sonnet-20241022", True, "anthropic")
    CLAUDE_3_5_HAIKU_20241022 = ("claude-3-5-haiku-20241022", True, "anthropic")

    # Gemini https://ai.google.dev/gemini-api/docs/models/gemini#model-variations
    # https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/call-gemini-using-openai-library#supported_models
    GEMINI_2_5_FLASH_20250417 = ("gemini-2.5-flash-preview", True, "google")
    GEMINI_2_5_PRO_EXP = ("gemini-2.5-pro-exp-03-25", True, "google")
    GEMINI_2_0_FLASH_THINKING_EXP = ("gemini-2.0-flash-thinking-exp-01-21", True, "google")
    GEMINI_2_0_FLASH = ("gemini-2.0-flash", True, "google")
    GEMINI_1_5_FLASH = ("gemini-1.5-flash", True, "google")
    GEMINI_EXP_1206 = ("gemini-exp-1206", True, "google")
    GEMINI_1_5_PRO = ("gemini-1.5-pro", True, "google")

    # OpenAI https://platform.openai.com/docs/models/overview
    O3 = ("o3", True, "openai")
    GPT_41 = ("gpt-4.1", True, "openai")
    GPT_41_MINI = ("gpt-4.1-mini", True, "openai")
    GPT_41_MINI_2025_04_14 = ("gpt-4.1-mini-2025-04-14", True, "openai")
    GPT_41_NANO = ("gpt-4.1-nano", True, "openai")
    GPT_41_NANO_2025_04_14 = ("gpt-4.1-nano-2025-04-14", True, "openai")
    O4_MINI = ("o4-mini", True, "openai")
    O4_MINI_2025_04_16 = ("o4-mini-2025-04-16", True, "openai")
    GPT_4O = ("gpt-4o", True, "openai")
    GPT_4O_MINI = ("gpt-4o-mini", True, "openai")
    GPT_45 = ("gpt-4.5-preview", True, "openai")
    O1 = ("o1", True, "openai")
    O1_MINI = ("o1-mini", False, "openai")
    O1_PREVIEW = ("o1-preview", False, "openai")
    O3_MINI = ("o3-mini", True, "openai")

    DEEP_SEEK_R1_7B = ("deepseek-r1:7b", False, "ollama")

    # Claude Code
    CLAUDE_CODE = ("claude-code", True, "claude_code")

    def __init__(self, id, tool_use_support, provider):
        self.id = id
        self.tool_use_support = tool_use_support
        self.provider = provider

    @property
    def model_id(self) -> str:
        return self.id

    @property
    def api_key_env(self) -> str:
        return f"{self.provider.upper()}_API_KEY"

    @classmethod
    def from_model_id(cls, model_id: str) -> "ChatModel":
        for model in cls:
            if model.model_id == model_id:
                return model
        raise ValueError(f"Model with id '{model_id}' not found.")

    @classmethod
    def get_api_key(self) -> str:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise ValueError(f"API key for {self.model.api_key_env} not found in environment variables")
        return api_key
