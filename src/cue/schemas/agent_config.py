from typing import Any, Optional
from pathlib import Path

from pydantic import Field, BaseModel

from .feature_flag import FeatureFlag


class AgentConfig(BaseModel):
    id: Optional[str] = "default_id"
    name: Optional[str] = "default_name"
    client_id: Optional[str] = "default_client"
    is_primary: Optional[bool] = False
    feedback_path: Optional[Path] = None
    project_context_path: Optional[str] = None
    # Detailed description of agent's role, capabilities, and collaboration patterns, which is used for other agents info
    description: Optional[str] = None
    # System message defining agent's behavior, collaboration guidelines, and boundaries
    instruction: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    use_cue: Optional[bool] = False
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    temperature: Optional[float] = 0.8
    max_tokens: Optional[int] = 5000
    max_context_tokens: Optional[int] = 20000
    memory_tokens: Optional[int] = 2000  # maximum memory token for each request
    stop_sequences: Optional[list[str]] = None
    tools: Optional[list[Any]] = []  # callable or tool enum
    enable_mcp: bool = False
    parallel_tool_calls: bool = True
    conversation_id: Optional[str] = None
    enable_services: bool = False
    feature_flag: FeatureFlag = Field(default_factory=FeatureFlag)
