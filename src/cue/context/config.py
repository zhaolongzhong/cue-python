"""Configuration for context management system."""

from dataclasses import dataclass

from ..schemas import FeatureFlag


@dataclass
class ContextConfig:
    """Configuration for DynamicContextManager.
    
    Keeps all configuration in one place for better maintainability
    and makes it easier to modify behavior without changing code.
    """
    model: str
    max_tokens: int
    feature_flag: FeatureFlag
    batch_remove_percentage: float = 0.30
    max_summaries: int = 6

    # Token management settings
    min_tokens_to_keep: int = 1000  # Minimum tokens to preserve
    excess_threshold: float = 0.25  # Maximum excess tokens before forced truncation

    def __post_init__(self):
        """Validate configuration values."""
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not 0 < self.batch_remove_percentage < 1:
            raise ValueError("batch_remove_percentage must be between 0 and 1")
        if self.min_tokens_to_keep >= self.max_tokens:
            raise ValueError("min_tokens_to_keep must be less than max_tokens")
