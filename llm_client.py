import os
import json
import requests
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Message:
    role: str
    content: str


class OpenRouterClient:
    def __init__(self, api_key: Optional[str] = None):
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
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"OpenRouter API error: {response.status_code} - {response.text}")
        
        return response.json()
    
    def get_completion_text(self, response: Dict) -> str:
        """Extract the text content from a chat completion response."""
        return response["choices"][0]["message"]["content"]
    
    def create_message(self, role: str, content: str) -> Message:
        """Helper to create a Message object."""
        return Message(role=role, content=content)