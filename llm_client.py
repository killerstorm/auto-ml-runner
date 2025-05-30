import os
import json
import requests
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from dotenv import load_dotenv
from datetime import datetime
import uuid
from pathlib import Path
from config import LoggingConfig

load_dotenv()


@dataclass
class Message:
    role: str
    content: str


class OpenRouterClient:
    def __init__(
        self, 
        api_key: Optional[str] = None,
        enable_logging: Optional[bool] = None,
        log_dir: Optional[str] = None
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable.")
        
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/auto-ml-runner",
            "X-Title": "Auto ML Runner"
        }
        
        # Logging configuration - use config.py defaults if not specified
        logging_config = LoggingConfig()
        self.enable_logging = enable_logging if enable_logging is not None else logging_config.enable_logging
        self.log_dir = Path(log_dir if log_dir is not None else logging_config.log_dir)
        
        if self.enable_logging:
            self.log_dir.mkdir(exist_ok=True)
    
    def _log_interaction(
        self,
        request_id: str,
        request_data: Dict,
        response_data: Optional[Dict] = None,
        error: Optional[str] = None
    ):
        """Log LLM interaction to a JSON file."""
        if not self.enable_logging:
            return
        
        timestamp = datetime.now().isoformat()
        log_entry = {
            "request_id": request_id,
            "timestamp": timestamp,
            "request": request_data,
            "response": response_data,
            "error": error
        }
        
        # Create filename with timestamp and request ID
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{request_id}.json"
        log_path = self.log_dir / filename
        
        with open(log_path, 'w') as f:
            json.dump(log_entry, f, indent=2)
    
    def chat_completion(
        self,
        messages: List[Union[Message, Dict[str, str]]],
        model: str,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict] = None,
        **kwargs
    ) -> Dict:
        """
        Send a chat completion request to OpenRouter.
        
        Args:
            messages: List of messages (Message objects or dicts with 'role' and 'content')
            model: Model identifier (e.g., 'anthropic/claude-3.5-sonnet')
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            response_format: Optional response format (e.g., {"type": "json_object"})
            **kwargs: Additional parameters to pass to the API
        
        Returns:
            API response as a dictionary
        """
        request_id = str(uuid.uuid4())[:8]  # Short ID for easier reference
        
        # Convert Message objects to dicts
        message_dicts = []
        for msg in messages:
            if isinstance(msg, Message):
                message_dicts.append({"role": msg.role, "content": msg.content})
            else:
                message_dicts.append(msg)
        
        payload = {
            "model": model,
            "messages": message_dicts,
            "temperature": temperature,
            **kwargs
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        if response_format:
            payload["response_format"] = response_format
        
        # Log the request
        request_data = {
            "url": f"{self.base_url}/chat/completions",
            "payload": payload,
            "headers": {k: v for k, v in self.headers.items() if k != "Authorization"}  # Don't log API key
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload
            )
            
            if response.status_code != 200:
                error_msg = f"OpenRouter API error: {response.status_code} - {response.text}"
                self._log_interaction(request_id, request_data, error=error_msg)
                raise Exception(error_msg)
            
            response_data = response.json()
            
            # Log successful interaction
            self._log_interaction(request_id, request_data, response_data)
            
            return response_data
            
        except Exception as e:
            # Log failed interaction
            self._log_interaction(request_id, request_data, error=str(e))
            raise
    
    def get_completion_text(self, response: Dict) -> str:
        """Extract the text content from a chat completion response."""
        return response["choices"][0]["message"]["content"]
    
    def create_message(self, role: str, content: str) -> Message:
        """Helper to create a Message object."""
        return Message(role=role, content=content)