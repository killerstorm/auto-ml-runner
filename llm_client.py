import os
import json
import requests
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from dotenv import load_dotenv
from datetime import datetime
import uuid
from pathlib import Path
import time
from config import LoggingConfig
import inspect
import traceback

load_dotenv()


@dataclass
class Message:
    role: str
    content: str


@dataclass
class CallContext:
    """Context about where the LLM call is coming from."""
    function_name: str
    run_number: Optional[int] = None
    experiment_id: Optional[str] = None
    iteration: Optional[int] = None
    task_id: Optional[str] = None
    module: Optional[str] = None
    additional_context: Optional[Dict] = None


class OpenRouterClient:
    def __init__(
        self, 
        api_key: Optional[str] = None,
        enable_logging: Optional[bool] = None,
        log_dir: Optional[str] = None,
        max_retries: int = 3,
        experiment_id: Optional[str] = None
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
        self.max_retries = max_retries
        self.experiment_id = experiment_id
        
        if self.enable_logging:
            self.log_dir.mkdir(exist_ok=True)
            # Create a structured log file for this session
            self.session_log_file = self.log_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    
    def _get_caller_context(self) -> Dict[str, str]:
        """Extract context about the calling function."""
        # Go up the stack to find the actual caller (not _log_interaction)
        stack = inspect.stack()
        caller_info = {}
        
        # Skip our own methods
        for i, frame_info in enumerate(stack):
            if frame_info.function not in ['_log_interaction', 'chat_completion', '_get_caller_context']:
                caller_info = {
                    'function': frame_info.function,
                    'module': frame_info.filename.split('/')[-1] if frame_info.filename else 'unknown',
                    'line': frame_info.lineno
                }
                break
        
        return caller_info
    
    def _log_interaction(
        self,
        request_id: str,
        request_data: Dict,
        response_data: Optional[Dict] = None,
        error: Optional[str] = None,
        context: Optional[CallContext] = None
    ):
        """Log LLM interaction to both individual JSON and session JSONL file."""
        if not self.enable_logging:
            return
        
        timestamp = datetime.now()
        
        # Get automatic caller context if not provided
        caller_info = self._get_caller_context()
        
        log_entry = {
            "request_id": request_id,
            "timestamp": timestamp.isoformat(),
            "experiment_id": self.experiment_id,
            "context": {
                "function_name": context.function_name if context else caller_info.get('function', 'unknown'),
                "run_number": context.run_number if context else None,
                "iteration": context.iteration if context else None,
                "task_id": context.task_id if context else None,
                "module": context.module if context else caller_info.get('module', 'unknown'),
                "line": caller_info.get('line'),
                "additional": context.additional_context if context else None
            },
            "request": request_data,
            "response": response_data,
            "error": error,
            "performance": {
                "duration_ms": None,  # Will be calculated later
                "tokens_used": response_data.get("usage", {}) if response_data else None
            }
        }
        
        # Create individual file (for backwards compatibility)
        filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{request_id}.json"
        log_path = self.log_dir / filename
        
        with open(log_path, 'w') as f:
            json.dump(log_entry, f, indent=2)
        
        # Append to session log file (JSONL format for streaming)
        with open(self.session_log_file, 'a') as f:
            json.dump(log_entry, f)
            f.write('\n')
    
    def _attempt_json_repair(self, content: str) -> Optional[str]:
        """
        Attempt to repair common JSON issues.
        Returns repaired JSON string or None if repair failed.
        """
        if not content:
            return None
        
        try:
            # First, try to parse as-is
            json.loads(content)
            return content
        except json.JSONDecodeError:
            pass
        
        # Try common fixes
        repaired = content
        
        # Fix missing closing braces/brackets at the end
        open_braces = repaired.count('{') - repaired.count('}')
        open_brackets = repaired.count('[') - repaired.count(']')
        
        if open_braces > 0:
            repaired += '}' * open_braces
        if open_brackets > 0:
            repaired += ']' * open_brackets
        
        # Try to parse the repaired version
        try:
            json.loads(repaired)
            return repaired
        except json.JSONDecodeError:
            pass
        
        # If content ends mid-string, try to close it
        if repaired.rstrip().endswith('"') == False and '"' in repaired:
            # Count quotes to see if we have an unclosed string
            quote_count = repaired.count('"') - repaired.count('\\"')
            if quote_count % 2 == 1:
                repaired += '"'
                
                # Now add closing braces/brackets again
                open_braces = repaired.count('{') - repaired.count('}')
                open_brackets = repaired.count('[') - repaired.count(']')
                
                if open_braces > 0:
                    repaired += '}' * open_braces
                if open_brackets > 0:
                    repaired += ']' * open_brackets
                
                try:
                    json.loads(repaired)
                    return repaired
                except json.JSONDecodeError:
                    pass
        
        return None
    
    def chat_completion(
        self,
        messages: List[Union[Message, Dict[str, str]]],
        model: str,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict] = None,
        context: Optional[CallContext] = None,
        **kwargs
    ) -> Dict:
        """
        Send a chat completion request to OpenRouter with retry logic.
        
        Args:
            messages: List of messages (Message objects or dicts with 'role' and 'content')
            model: Model identifier (e.g., 'anthropic/claude-3.5-sonnet')
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            response_format: Optional response format (e.g., {"type": "json_object"})
            context: Optional context about where this call is coming from
            **kwargs: Additional parameters to pass to the API
        
        Returns:
            API response as a dictionary
        """
        request_id = str(uuid.uuid4())[:8]  # Short ID for easier reference
        start_time = time.time()
        
        # Convert Message objects to dicts
        message_dicts = []
        for msg in messages:
            if isinstance(msg, Message):
                message_dicts.append({"role": msg.role, "content": msg.content})
            else:
                message_dicts.append(msg)
        
        # Track the original max_tokens for retries
        original_max_tokens = max_tokens
        
        # Set a reasonable default for JSON responses to prevent truncation
        if response_format and response_format.get("type") in ["json_object", "json_schema"]:
            if not max_tokens:
                max_tokens = 4000  # Default for JSON responses
                self._log_interaction(
                    request_id,
                    {"note": "Setting default max_tokens=4000 for JSON response"},
                    None,
                    None
                )
        
        for attempt in range(self.max_retries):
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
            
            # Log the request with context
            request_data = {
                "url": f"{self.base_url}/chat/completions",
                "payload": payload,
                "headers": {k: v for k, v in self.headers.items() if k != "Authorization"},  # Don't log API key
                "model": model,
                "attempt": attempt + 1
            }
            
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload
                )
                
                if response.status_code != 200:
                    error_msg = f"OpenRouter API error: {response.status_code} - {response.text}"
                    self._log_interaction(request_id, request_data, error=error_msg, context=context)
                    
                    # Check if it's a rate limit error and we should retry
                    if response.status_code == 429 and attempt < self.max_retries - 1:
                        retry_after = int(response.headers.get('Retry-After', 5))
                        time.sleep(retry_after)
                        continue
                    
                    raise Exception(error_msg)
                
                response_data = response.json()
                
                # Check for zero-length response
                content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if not content:
                    error_msg = "Received zero-length response from API"
                    self._log_interaction(request_id, request_data, response_data, error=error_msg, context=context)
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    raise Exception(error_msg)
                
                # Get finish reason for later use
                finish_reason = response_data.get("choices", [{}])[0].get("finish_reason", "")
                
                # Check if JSON parsing is needed and valid
                if response_format and response_format.get("type") in ["json_object", "json_schema"]:
                    try:
                        json.loads(content)
                    except json.JSONDecodeError as e:
                        # First, try to repair the JSON
                        repaired_content = self._attempt_json_repair(content)
                        if repaired_content:
                            # Update the response with repaired content
                            response_data["choices"][0]["message"]["content"] = repaired_content
                            self._log_interaction(
                                request_id,
                                request_data,
                                response_data,
                                error=f"JSON repaired successfully (original error: {str(e)})",
                                context=context
                            )
                            return response_data
                        
                        error_msg = f"Invalid JSON in response: {str(e)}"
                        self._log_interaction(request_id, request_data, response_data, error=error_msg, context=context)
                        
                        if attempt < self.max_retries - 1:
                            # Check if this might be a truncation issue
                            error_str = str(e).lower()
                            is_truncation = any(term in error_str for term in [
                                "unterminated string",
                                "expecting value",
                                "unexpected end",
                                "expecting property name",
                                "expecting ',' delimiter"
                            ])
                            
                            # Also check if finish_reason suggests truncation
                            is_length_issue = finish_reason == "length" or (
                                is_truncation and content.rstrip()[-1] not in ['}', ']']
                            )
                            
                            if is_length_issue:
                                # Increase max_tokens for likely truncation issues
                                if max_tokens:
                                    max_tokens = int(max_tokens * 1.5)
                                else:
                                    # If no max_tokens was set, start with a reasonable default
                                    max_tokens = 4000
                                
                                self._log_interaction(
                                    request_id,
                                    request_data,
                                    response_data,
                                    error=f"JSON likely truncated, retrying with max_tokens={max_tokens}",
                                    context=context
                                )
                            
                            time.sleep(2 ** attempt)  # Exponential backoff
                            continue
                        raise Exception(error_msg)
                
                # Check if stopped due to length limit (non-JSON responses)
                if finish_reason == "length" and max_tokens and attempt < self.max_retries - 1:
                    # Skip if we already handled this in JSON parsing above
                    if not (response_format and response_format.get("type") in ["json_object", "json_schema"]):
                        # Retry with 50% more tokens
                        max_tokens = int(max_tokens * 1.5)
                        self._log_interaction(
                            request_id, 
                            request_data, 
                            response_data, 
                            error=f"Hit length limit, retrying with max_tokens={max_tokens}",
                            context=context
                        )
                        time.sleep(1)  # Brief pause before retry
                        continue
                
                # Log successful interaction with timing
                duration_ms = int((time.time() - start_time) * 1000)
                if 'performance' not in request_data:
                    request_data['performance'] = {}
                request_data['performance']['duration_ms'] = duration_ms
                self._log_interaction(request_id, request_data, response_data, context=context)
                
                return response_data
                
            except requests.exceptions.RequestException as e:
                error_msg = f"Request error: {str(e)}"
                self._log_interaction(request_id, request_data, error=error_msg, context=context)
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                raise
            except Exception as e:
                # For other exceptions, check if we should retry
                if attempt < self.max_retries - 1:
                    # Only retry on specific errors
                    error_str = str(e).lower()
                    if any(retry_term in error_str for retry_term in ["zero-length", "invalid json", "timeout"]):
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                # Log and re-raise if not retrying
                self._log_interaction(request_id, request_data, error=str(e), context=context)
                raise
        
        # If we've exhausted all retries
        final_error = f"Failed after {self.max_retries} attempts"
        self._log_interaction(request_id, request_data, error=final_error, context=context)
        raise Exception(final_error)
    
    def get_completion_text(self, response: Dict) -> str:
        """Extract the text content from a chat completion response."""
        return response["choices"][0]["message"]["content"]
    
    def create_message(self, role: str, content: str) -> Message:
        """Helper to create a Message object."""
        return Message(role=role, content=content)