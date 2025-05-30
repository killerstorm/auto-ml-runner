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
    tasks_model: Optional[str] = None
    
    def __post_init__(self):
        # Override with environment variables if present
        self.summarize_model = os.getenv("SUMMARIZE_MODEL", self.summarize_model)
        self.code_model = os.getenv("CODE_MODEL", self.code_model)
        self.report_model = os.getenv("REPORT_MODEL", self.report_model)
        self.plan_model = os.getenv("PLAN_MODEL", self.plan_model or self.code_model)
        self.tasks_model = os.getenv("TASKS_MODEL", self.tasks_model or self.summarize_model)


@dataclass
class RunConfig:
    """Configuration for experiment runs."""
    max_runs: int = 10
    max_retries: int = 3
    timeout_seconds: int = 7200  # 2 hours
    temperature: float = 0.7
    max_tokens_code: int = 8000
    max_tokens_summary: int = 2000
    max_tokens_report: int = 4000
    
    def __post_init__(self):
        # Override with environment variables if present
        self.max_runs = int(os.getenv("MAX_RUNS", self.max_runs))
        self.max_retries = int(os.getenv("MAX_RETRIES", self.max_retries))
        self.timeout_seconds = int(os.getenv("TIMEOUT_SECONDS", self.timeout_seconds))
        self.temperature = float(os.getenv("TEMPERATURE", self.temperature))