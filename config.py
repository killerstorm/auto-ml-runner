import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ModelConfig:
    """Configuration for models used in different stages."""
    summarize_model: str = "openai/gpt-4o-mini"
    code_model: str = "anthropic/claude-3.5-sonnet"
    report_model: str = "anthropic/claude-3.5-sonnet"
    plan_model: Optional[str] = None
    analysis_model: Optional[str] = None
    
    def __post_init__(self):
        # Override with environment variables if present
        self.summarize_model = os.getenv("SUMMARIZE_MODEL", self.summarize_model)
        self.code_model = os.getenv("CODE_MODEL", self.code_model)
        self.report_model = os.getenv("REPORT_MODEL", self.report_model or self.code_model)
        self.plan_model = os.getenv("PLAN_MODEL", self.plan_model or self.code_model)
        self.analysis_model = os.getenv("ANALYSIS_MODEL", self.analysis_model or self.code_model)


@dataclass
class RunConfig:
    """Configuration for experiment runs."""
    max_runs: int = 10
    max_retries: int = 3
    timeout_seconds: int = 7200  # 2 hours
    temperature: float = 0.3
    max_tokens_code: int = 30000
    max_tokens_summary: int = 5000
    max_tokens_report: int = 30000
    max_tokens_analysis: int = 30000
    
    def __post_init__(self):
        # Override with environment variables if present
        self.max_runs = int(os.getenv("MAX_RUNS", self.max_runs))
        self.max_retries = int(os.getenv("MAX_RETRIES", self.max_retries))
        self.timeout_seconds = int(os.getenv("TIMEOUT_SECONDS", self.timeout_seconds))
        self.temperature = float(os.getenv("TEMPERATURE", self.temperature))


@dataclass
class LoggingConfig:
    """Configuration for LLM interaction logging."""
    enable_logging: bool = False
    log_dir: str = "llm_logs"
    
    def __post_init__(self):
        # Override with environment variables if present
        self.enable_logging = os.getenv("LLM_ENABLE_LOGGING", "false").lower() in ("true", "1", "yes")
        self.log_dir = os.getenv("LLM_LOG_DIR", self.log_dir)