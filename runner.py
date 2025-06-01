#!/usr/bin/env python3
import os
import sys
import json
import time
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json
import click
from rich.console import Console
from rich.markdown import Markdown

from llm_client import OpenRouterClient, Message, CallContext
from config import ModelConfig, RunConfig, LoggingConfig
from console_utils import (
    log_info, log_success, log_error, log_warning, 
    log_progress, log_section, log_panel, log_status
)
from task_manager import TaskManager
from schemas import ( 
    INITIAL_TASKS_SCHEMA, ANALYSIS_ONLY_SCHEMA, TASK_UPDATES_SCHEMA
)

console = Console()

# Simple wrapper using console.status for reliability
class Spinner:
    """Context manager for progress spinners."""
    def __init__(self, description: str):
        self.description = description
        self.status = None
    
    def __enter__(self):
        self.status = console.status(self.description, spinner="dots")
        self.status.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.status:
            try:
                self.status.stop()
            except:
                pass

SYSTEM_PROMPT_CONTEXT = "We are conducting automated ML experiments.\n "

def system_prompt(role: str) -> str:
    return f"""
{SYSTEM_PROMPT_CONTEXT}
Your role is {role}
"""

main_py = "main.py"

def get_environment_info() -> Dict[str, any]:
    """Get comprehensive environment information."""
    env_info = {
        "python_version": sys.version,
        "platform": sys.platform,
        "gpu_available": False,
        "gpu_info": None,
        "torch_version": None,
        "cuda_version": None,
        "installed_packages": []
    }
    
    # Get GPU information
    try:
        import torch
        env_info["torch_version"] = torch.__version__
        
        if torch.cuda.is_available():
            env_info["gpu_available"] = True
            env_info["cuda_version"] = torch.version.cuda
            
            # Get GPU details
            gpu_props = torch.cuda.get_device_properties(0)
            env_info["gpu_info"] = {
                "name": torch.cuda.get_device_name(0),
                "memory_gb": gpu_props.total_memory / 1e9,
                "compute_capability": f"{gpu_props.major}.{gpu_props.minor}",
                "multi_processor_count": gpu_props.multi_processor_count
            }
    except ImportError:
        pass
    except Exception as e:
        env_info["gpu_error"] = str(e)
    
    # Get installed packages
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            packages = json.loads(result.stdout)
            env_info["installed_packages"] = [
                f"{pkg['name']}=={pkg['version']}" 
                for pkg in packages 
            ]
    except Exception as e:
        env_info["package_error"] = str(e)
    
    return env_info

def restrict_text(text: str, max_length: int, max_lines: int, remove_middle: bool = True) -> str:
    """
    Restrict text to a maximum length and number of lines by intelligently truncating.
    
    Args:
        text: Input text to restrict
        max_length: Maximum number of characters allowed
        max_lines: Maximum number of lines allowed
        remove_middle: If True, remove the middle part of text, otherwise remove starting lines
    
    Returns:
        Truncated text string
    """
    # Split into lines and handle line count restriction
    lines = text.splitlines()
    
    if len(lines) > max_lines:
        if remove_middle:
            # Keep equal parts from beginning and end
            keep_start = max_lines // 2
            keep_end = max_lines - keep_start - 1  # -1 for the ellipsis line
            if keep_start > 0 and keep_end > 0:
                lines = lines[:keep_start] + ["... (truncated middle) ..."] + lines[-keep_end:]
            else:
                lines = lines[:max_lines]
        else:
            # Remove from the beginning
            lines = ["... (truncated beginning) ..."] + lines[-(max_lines-1):]
    
    # Rejoin lines
    text = "\n".join(lines)
    
    # Handle character length restriction
    if len(text) > max_length:
        if remove_middle:
            # Calculate how much to keep from start and end
            keep_chars = max_length - 30  # Reserve space for truncation marker
            keep_start = keep_chars // 2
            keep_end = keep_chars - keep_start
            
            if keep_start > 0 and keep_end > 0:
                text = text[:keep_start] + "\n... (truncated) ...\n" + text[-keep_end:]
            else:
                text = text[:max_length-3] + "..."
        else:
            # Remove from the beginning
            text = "... (truncated) ..." + text[-(max_length-20):]
            
    return text

def extract_key_log_info(logs: str) -> Dict[str, str]:
    """Extract critical information from logs before summarization."""
    lines = logs.splitlines()
    extracted = {
        'errors': [],
        'warnings': [],
        'final_metrics': [],
        'gpu_memory': []
    }
    
    # Look for common patterns in ML logs
    for i, line in enumerate(lines):
        line_lower = line.lower()
        
        # Capture errors and their context
        if 'error' in line_lower or 'exception' in line_lower or 'traceback' in line_lower:
            # Get some context around errors
            start = max(0, i - 2)
            end = min(len(lines), i + 5)
            error_context = '\n'.join(lines[start:end])
            if error_context not in '\n'.join(extracted['errors']):
                extracted['errors'].append(error_context)
        
        # Capture warnings
        elif 'warning' in line_lower:
            extracted['warnings'].append(line.strip())
        
        # Look for final metrics (common patterns)
        elif any(keyword in line_lower for keyword in ['final', 'test', 'eval', 'validation']) and \
             any(metric in line_lower for metric in ['loss', 'accuracy', 'f1', 'precision', 'recall', 'bleu', 'perplexity']):
            extracted['final_metrics'].append(line.strip())
        
        # GPU memory usage
        elif 'gpu' in line_lower and ('memory' in line_lower or 'mem' in line_lower):
            extracted['gpu_memory'].append(line.strip())
    
    # Also look for metrics at the end of the log
    last_lines = lines[-20:] if len(lines) > 20 else lines
    for line in last_lines:
        if any(char in line for char in [':', '=', '|']) and \
           any(word in line.lower() for word in ['loss', 'accuracy', 'epoch', 'step']):
            if line.strip() not in extracted['final_metrics']:
                extracted['final_metrics'].append(line.strip())
    
    return extracted

def get_code_changes(old_code: Optional[str], new_code: str) -> str:
    """Get a simple summary of code changes between runs."""
    if not old_code:
        return "Initial implementation"
    
    old_lines = old_code.splitlines()
    new_lines = new_code.splitlines()
    
    # Simple line count comparison
    diff_summary = f"Lines changed: {len(old_lines)} → {len(new_lines)} ({len(new_lines) - len(old_lines):+d})\n"
    
    # Find major structural changes
    old_imports = [l for l in old_lines if l.strip().startswith(('import ', 'from '))]
    new_imports = [l for l in new_lines if l.strip().startswith(('import ', 'from '))]
    
    added_imports = set(new_imports) - set(old_imports)
    if added_imports:
        diff_summary += f"New imports: {', '.join(i.split()[-1] for i in list(added_imports)[:3])}\n"
    
    # Look for new function/class definitions
    old_defs = [l.strip() for l in old_lines if l.strip().startswith(('def ', 'class '))]
    new_defs = [l.strip() for l in new_lines if l.strip().startswith(('def ', 'class '))]
    
    added_defs = set(new_defs) - set(old_defs)
    if added_defs:
        diff_summary += f"New functions/classes: {len(added_defs)}\n"
    
    return diff_summary.strip()

class ExperimentRunner:
    def __init__(self, experiment_dir: Path, config: RunConfig, model_config: ModelConfig):
        self.experiment_dir = experiment_dir
        self.config = config
        self.model_config = model_config
        
        # Generate a unique experiment ID
        self.experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{experiment_dir.name}"
        
        log_config = LoggingConfig()
        if log_config.enable_logging:
            log_dir = Path(log_config.log_dir)
            if not log_dir.is_absolute():
                log_dir = self.experiment_dir / log_dir
            self.client = OpenRouterClient(
                log_dir=log_dir, 
                max_retries=self.config.max_retries,
                experiment_id=self.experiment_id
            )
        else:
            self.client = OpenRouterClient(
                max_retries=self.config.max_retries,
                experiment_id=self.experiment_id
            )
        
        # Create experiment directory if it doesn't exist
        self.experiment_dir.mkdir(exist_ok=True)
        
        # Create shared files directory for persistent data across runs
        self.shared_files_dir = self.experiment_dir / "shared_files"
        self.shared_files_dir.mkdir(exist_ok=True)
        
        # Initialize state files
        self.plan_file = self.experiment_dir / "PLAN.md"
        self.tasks_file = self.experiment_dir / "tasks.json"
        self.findings_file = self.experiment_dir / "EXPERIMENT_LOG.md"
        self.idea_file = self.experiment_dir / "IDEA.md"
        
        # Initialize task manager
        self.task_manager = TaskManager(self.tasks_file)
    
    def get_current_run_number(self) -> int:
        """Find the highest run number in the experiment directory."""
        run_dirs = [d for d in self.experiment_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
        if not run_dirs:
            return 0
        
        run_numbers = []
        for d in run_dirs:
            try:
                run_numbers.append(int(d.name.split("_")[1]))
            except (IndexError, ValueError):
                continue
        
        return max(run_numbers) if run_numbers else 0
    
    def create_run_directory(self, run_number: int) -> Path:
        """Create a directory for a specific run."""
        run_dir = self.experiment_dir / f"run_{run_number}"
        run_dir.mkdir(exist_ok=True)
        return run_dir
    
    def read_file_safe(self, file_path: Path) -> str:
        """Read file content, return empty string if file doesn't exist."""
        if file_path.exists():
            return file_path.read_text()
        return ""
    
    def get_previous_code(self, run_number: int) -> Optional[str]:
        """Get the code from the previous run."""
        if run_number <= 1:
            return None
        
        prev_run_dir = self.experiment_dir / f"run_{run_number - 1}"
        prev_code_file = prev_run_dir / main_py
        
        if prev_code_file.exists():
            return prev_code_file.read_text()
        else:
            return self.get_previous_code(run_number - 1)
    
    def get_logs(self, run_number: int) -> Optional[str]:
        """Get the logs from the previous run."""
        if run_number <= 1:
            return None
        
        prev_run_dir = self.experiment_dir / f"run_{run_number}"
        stdout_file = prev_run_dir / "stdout.txt"
        stderr_file = prev_run_dir / "stderr.txt"
        
        # Read both stdout and stderr if they exist
        stdout_content = stdout_file.read_text() if stdout_file.exists() else ""
        stderr_content = stderr_file.read_text() if stderr_file.exists() else ""
        
        if stdout_content or stderr_content:
            return f"STDOUT:\n{stdout_content}\n\nSTDERR:\n{stderr_content}"
        else:
            return ""
    
    def get_or_create_log_summary(self, run_number: int, logs: Optional[str] = None) -> Optional[str]:
        """Get existing log summary or create a new one if needed.
        
        Args:
            run_number: The run number to get/create summary for
            logs: Optional log content if summary needs to be created
            
        Returns:
            Log summary string, or None if no logs available
        """
        run_dir = self.experiment_dir / f"run_{run_number}"
        summary_file = run_dir / "summary.txt"
        
        # Check if summary already exists
        if summary_file.exists():
            return summary_file.read_text()
        
        # If no logs provided, try to read from stdout and stderr files
        if logs is None:
            logs = self.get_logs(run_number)
        
        # Generate new summary
        summary = self.summarize_logs(logs, run_number)
        
        # Save summary to file
        summary_file.write_text(summary)
        
        return summary
    
    def summarize_logs(self, logs: str, run_number: int) -> str:
        """Use LLM to create a mechanical summary of the logs."""
        
        # Restrict logs to reasonable size
        rlogs = restrict_text(logs, 1500000, 15000)
        
        messages = [
            Message("system", system_prompt("""an assistant extracting key information from ML experiment logs.
Focus on:
- Errors and warnings
- Training metrics (loss, accuracy, etc.)
- Completion status
- Hardware utilization
- Any anomalies or issues""")),
            
            Message("user", f"""Summarize the key information from run {run_number} logs.

FULL LOGS (truncated):
------
{rlogs}
------

Extract and copy important information from training logs.
Do NOT interpret or analyze - just extract factual information.

These warnings are know to be irrelevant: and should be ignored:
"Unable to register cuFFT factory"
"computation_placer.cc"
"cpu_feature_guard.cc"

""")
        ]
        
        # Create context for this call
        context = CallContext(
            function_name="summarize_logs",
            run_number=run_number,
            experiment_id=self.experiment_id
        )
        
        response = self.client.chat_completion(
            messages=messages,
            model=self.model_config.summarize_model,
            temperature=0.1,  # Low temperature for factual extraction
            max_tokens=self.config.max_tokens_summary,
            context=context
        )
        
        return self.client.get_completion_text(response)
    
    def generate_code(self, run_number: int, previous_code: Optional[str], log_summary: Optional[str]) -> Tuple[List[str], str]:
        """Generate code for the next run."""
        idea = self.read_file_safe(self.idea_file)
        plan = self.read_file_safe(self.plan_file)
        tasks = self.task_manager.to_markdown()
        findings = self.read_file_safe(self.findings_file)
        
        # Get environment information for code generation
        env_info = get_environment_info()
        env_context = f"""
Environment:
- GPU: {'Available' if env_info['gpu_available'] else 'Not available'}"""
        if env_info['gpu_info']:
            env_context += f" - {env_info['gpu_info']['name']} ({env_info['gpu_info']['memory_gb']:.1f}GB)"
        env_context += f"\n- PyTorch: {env_info['torch_version'] or 'Not installed'}"
        if env_info['installed_packages']:
            env_context += "\n- Key packages: " + ", ".join(env_info['installed_packages'][:10])
        
        system_prompt_text = system_prompt("""an expert ML engineer implementing experiments. Generate complete, runnable Python code based on the provided context.""")
        
        user_prompt = f"""Generate the complete {main_py} code for run {run_number}.

{env_context}

Important guidelines:
- Generate a complete, self-contained {main_py} file, not patches or fragments
- Write all information to standard out and standard error, which will be captured by the experiment runner for further analysis
  * Specifically write anything that might be useful - intermediate results, sizes, sampled data, etc.
- Do not make progress bars (tqdm, etc)
- Do not create any files unless it's necessary according to the plan. Data in files won't be analyzed. Focus all your effort on standard out and standard error.
- Make incremental improvements based on previous results
- Focus on achieving the experimental goals
- Consider the available hardware and installed packages

File Management:
- You can access files from the previous run using relative paths: ../run_{run_number - 1}/filename
- If your code produces files needed by future runs (e.g., model checkpoints), save them in the current directory

General Plan:
{plan}

Experiment log / key findings:
{findings}

Tasks Status:
{tasks}


"""
                
        if log_summary:
            user_prompt += f"""Log summary from previous run:
------
{log_summary}
------

"""
        if previous_code:
            user_prompt += f"""Previous Code:
```python
{previous_code}
```

"""

        
        user_prompt += """Based on the above context, generate an improved version of main.py that addresses any issues and makes progress toward the experimental goals.


Response format:
Start your response with a list of task IDs you will be working on in this run, formatted as:

WORKING_ON_TASKS:
- task_001
- task_003

---

Then provide the Python code wrapped in ```python...```, no other explanations."""
        
        messages = [
            Message("system", system_prompt_text),
            Message("user", user_prompt)
        ]
        
        # Create context for this call
        context = CallContext(
            function_name="generate_code",
            run_number=run_number,
            experiment_id=self.experiment_id
        )
        
        response = self.client.chat_completion(
            messages=messages,
            model=self.model_config.code_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens_code,
            context=context
        )
        
        full_response = self.client.get_completion_text(response).strip()
        
        # Parse out the working tasks and code
        working_tasks = []
        code = full_response
        
        if "WORKING_ON_TASKS:" in full_response:
            parts = full_response.split("```python", 1)
            if len(parts) == 2:
                task_section = parts[0]
                code = parts[1].strip()
                
                # Extract task IDs
                lines = task_section.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('- '):
                        task_id = line[2:].strip()  # Remove "- "
                        working_tasks.append(task_id)
        
        # Clean up code formatting
        if code.startswith("```python"):
            code = code[9:]
        if code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        
        return working_tasks, code.strip()
    
    def revise_plan(self, reason: str):
        """Revise the experimental plan based on current findings."""
        current_plan = self.read_file_safe(self.plan_file)
        findings = self.read_file_safe(self.findings_file)
        task_state = self.task_manager.to_markdown()
        idea = self.read_file_safe(self.idea_file)
        
        # Get current environment information
        env_info = get_environment_info()
        env_summary = f"""
Current Environment:
- Python: {env_info['python_version'].split()[0]}
- Platform: {env_info['platform']}
- PyTorch: {env_info['torch_version'] or 'Not installed'}
- GPU: {'Available' if env_info['gpu_available'] else 'Not available'}
"""
        if env_info['gpu_info']:
            gpu = env_info['gpu_info']
            env_summary += f"  - {gpu['name']} ({gpu['memory_gb']:.1f}GB)\n"
        
        if env_info['installed_packages']:
            env_summary += "\nCurrently Installed Packages:\n"
            for pkg in sorted(env_info['installed_packages'])[:15]:
                env_summary += f"  - {pkg}\n"
        
        messages = [
            Message("system", system_prompt("""an expert ML researcher revising experimental plans. 
You should maintain the core objectives while adapting the approach based on learnings.""")),
            Message("user", f"""Revise the experimental plan based on current progress and findings.

Reason for revision: {reason}

Original idea:
------
{idea}
------

Current Plan:
------
{current_plan}
------

Experiment log / key findings:
------
{findings}
------

Current Task Status:
------
{task_state}
------

{env_summary}

Create a revised plan that:
1. Maintains the original experimental goals
2. Incorporates lessons learned
3. Adjusts methodology based on findings
4. Updates milestones and success criteria as needed
5. Addresses any blockers or challenges discovered - consider workarounds
6. Considers the current environment and available packages

Format as a structured markdown document.""")
        ]
        
        # Create context for this call - note we don't have run_number here
        context = CallContext(
            function_name="revise_plan",
            experiment_id=self.experiment_id,
            additional_context={"reason": reason}
        )
        
        response = self.client.chat_completion(
            messages=messages,
            model=self.model_config.plan_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens_code,
            context=context
        )
        
        revised_plan = self.client.get_completion_text(response)
        
        # Archive the old plan
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = self.experiment_dir / f"PLAN_archive_{timestamp}.md"
        shutil.copy(self.plan_file, archive_path)
        
        # Write the revised plan
        self.plan_file.write_text(revised_plan)
        
        # Add a note to findings about the plan revision
        revision_note = f"\n\n## Plan Revision - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        revision_note += f"**Reason:** {reason}\n\n"
        revision_note += "The experimental plan has been revised. See PLAN.md for the updated version.\n"
        
        with open(self.findings_file, "a") as f:
            f.write(revision_note)
        
        log_success(f"Revised experimental plan (archived old plan as {archive_path.name})")
    
    def analyze_run_results(self, run_number: int, log_summary: str, code: str, success: bool) -> Dict[str, any]:
        """Analyze the results of a run and return analysis, findings, and experiment state."""
        try:
            with Spinner("Analyzing run results..."):
                # Get current state
                current_findings = self.read_file_safe(self.findings_file)
                plan = self.read_file_safe(self.plan_file)
                
                # Get code changes summary
                previous_code = self.get_previous_code(run_number)
                code_changes = get_code_changes(previous_code, code)
                
                # Create analysis prompt
                messages = [
                    Message("system", system_prompt("""an expert ML engineer analyzing experiment results. 
Provide a structured JSON response with:
1. 'analysis': Your interpretation of the run results
2. 'key_findings': Key discoveries from this run (if any)
3. 'experiment_state': High-level experiment status flags""")),

                    Message("user", f"""Analyze the results from run {run_number}.

Experimental Plan:
------
{plan}
------

Code Changes Summary:
{code_changes}

Code from Run {run_number}:
------
{code}
------

Analysis of previous runs, findings, etc:
------
{restrict_text(current_findings, 50000, 2000, remove_middle=False) if current_findings else 'No previous findings'}
------

Log Summary from Run {run_number}:
------
{log_summary}
------

Analyze what happened and provide:
1. Analysis of the results from current run 
2. Key findings - at most 5 bullet points
3. Experiment state flags:
   - experiment_complete: Have we achieved the experimental goals?
   - plan_revision_needed: Should we revise the experimental plan based on current progress?
   - early_exit_required: Are there blockers preventing further progress?

Note:
    * We do not expect each run to be successful, as code written by your fellow AI might contain bugs, errors, etc.
      With simple bugs it is sufficient to document them in `analysis` section so they can be fixed the code generated on the next run.
      However, if there's more complex issue, you might suggest a workaround or a plan revision. It is entirely acceptable to
      dedicate a "run" to investigation/debugging - printing out all relevant information, etc.
    * Set `plan_revision_needed` to `false` if analysis already contains enough information to take a corrective action or improve results.
      It should set to `true` only if a signficiant correction is necessary.
    * Do not set `early_exit_required` to `true` if there's a possibility to fix the issue with a plan revision,
      using some workaround, etc.
""")
                ]
                
                # Create context for this call
                context = CallContext(
                    function_name="analyze_run_results",
                    run_number=run_number,
                    experiment_id=self.experiment_id,
                    additional_context={"success": success}
                )
                
                response = self.client.chat_completion(
                    messages=messages,
                    model=self.model_config.analysis_model,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens_analysis,
                    response_format={"type": "json_schema", "json_schema": {"name": "analysis", "schema": ANALYSIS_ONLY_SCHEMA}},
                    context=context
                )
                
                result = json.loads(self.client.get_completion_text(response))
                
                # Update EXPERIMENT_LOG.md with the analysis and key findings
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                finding = f"\n\n## Run {run_number} - {timestamp}\n\n### Analysis\n{result['analysis']}\n\n### Key Findings\n"
                for kf in result.get('key_findings', []):
                    finding += f"- {kf}\n"
                
                with open(self.findings_file, "a") as f:
                    f.write(finding)
                
                return result
                
        except Exception as e:
            log_error(f"Failed to analyze run results: {e}")
            # Fallback: save basic summary as finding
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            finding = f"\n\n## Run {run_number} - {timestamp}\n\n### Log Summary\n{log_summary}\n"
            with open(self.findings_file, "a") as f:
                f.write(finding)
            
            # Return default analysis
            return {
                'analysis': f"Error analyzing run: {str(e)}",
                'key_findings': ["Analysis failed - see log summary"],
                'experiment_state': {
                    'experiment_complete': False,
                    'plan_revision_needed': False,
                    'early_exit_required': False
                }
            }
    
    def update_tasks_from_analysis(self, run_number: int, analysis: Dict[str, any], log_summary: str, plan: str, code: str) -> None:
        """Update tasks based on the analysis results."""
        try:
            with Spinner("Updating task list..."):
                # Get current task state
                task_state = self.task_manager.get_structured_state()
                
                # Create task update prompt
                messages = [
                    Message("system", system_prompt("""a task manager for ML experiments. 
Based on the analysis of the latest run, determine what task updates are needed.""")),

                    Message("user", f"""Based on the analysis of run {run_number}, update the task list.

Experimental Plan:
------
{plan}
------

Code from Run {run_number}:
------
{code}
------

Analysis:
------
{analysis['analysis']}
------

Key Findings:
------
{json.dumps(analysis['key_findings'], indent=2)}
------

Log Summary:
------
{log_summary}
------

Current Tasks (Total: {task_state['total_tasks']}, Pending: {task_state['pending']}, In Progress: {task_state['in_progress']}, Completed: {task_state['completed']}):
{json.dumps(task_state['tasks'], indent=2)}

Determine task updates based on the analysis. You can:
- Mark tasks as complete (with notes explaining what was accomplished)
- Add new tasks discovered during the run
- Update task status to blocked if there are dependencies
- Remove tasks that are no longer relevant

Focus on actionable tasks that will help achieve the experimental goals.""")
                ]
                
                # Create context for this call
                context = CallContext(
                    function_name="update_tasks_from_analysis",
                    run_number=run_number,
                    experiment_id=self.experiment_id
                )
                
                response = self.client.chat_completion(
                    messages=messages,
                    model=self.model_config.analysis_model,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens_analysis,
                    response_format={"type": "json_schema", "json_schema": {"name": "task_updates", "schema": TASK_UPDATES_SCHEMA}},
                    context=context
                )
                
                result = json.loads(self.client.get_completion_text(response))
                
                # Process task updates
                for update in result.get('task_updates', []):
                    action = update.get('action')
                    
                    if action == 'complete':
                        self.task_manager.mark_completed(
                            update['task_id'], 
                            notes=update.get('notes')
                        )
                    elif action == 'add':
                        self.task_manager.add_task(
                            description=update['description'],
                            priority=update.get('priority', 'medium')
                        )
                    elif action == 'update':
                        updates = {'status': update.get('new_status')}
                        if 'notes' in update:
                            updates['notes'] = update['notes']
                        self.task_manager.update_task(update['task_id'], **updates)
                    elif action == 'remove':
                        # Actually remove the task to reduce clutter
                        self.task_manager.remove_task(update['task_id'])
                
        except Exception as e:
            log_error(f"Failed to update tasks: {e}")
            # Continue without task updates
    
    def run_training(self, run_dir: Path, code: str) -> Tuple[bool, str]:
        """Execute the training code and capture output."""
        code_file = run_dir / "main.py"
        stdout_file = run_dir / "stdout.txt"
        stderr_file = run_dir / "stderr.txt"
        
        # Write code to file
        code_file.write_text(code)
        
        # Run the training script
        log_progress(f"Running training script in {run_dir}")
        
        try:
            with open(stdout_file, 'w') as out, open(stderr_file, 'w') as err:
                result = subprocess.run(
                    [sys.executable, "main.py"],
                    cwd=run_dir,
                    stdout=out,
                    stderr=err,
                    text=True,
                    timeout=self.config.timeout_seconds
                )
            
            # Read the output files
            stdout_content = stdout_file.read_text() if stdout_file.exists() else ""
            stderr_content = stderr_file.read_text() if stderr_file.exists() else ""
            
            # Combine for return value
            output = f"STDOUT:\n{stdout_content}\n\nSTDERR:\n{stderr_content}"
            
            success = result.returncode == 0
            return success, output
            
        except subprocess.TimeoutExpired:
            # Read any partial output that was written before timeout
            stdout_content = stdout_file.read_text() if stdout_file.exists() else ""
            stderr_content = stderr_file.read_text() if stderr_file.exists() else ""
            
            # Append timeout message to stderr
            timeout_msg = f"\n\n[TIMEOUT] Training script timed out after {self.config.timeout_seconds} seconds\n"
            with open(stderr_file, 'a') as err:
                err.write(timeout_msg)
            
            # Combine output including the timeout message
            output = f"STDOUT:\n{stdout_content}\n\nSTDERR:\n{stderr_content}{timeout_msg}"
            return False, output
            
        except Exception as e:
            # Read any partial output
            stdout_content = stdout_file.read_text() if stdout_file.exists() else ""
            stderr_content = stderr_file.read_text() if stderr_file.exists() else ""
            
            # Append error message to stderr
            error_msg = f"\n\n[ERROR] Error running training script: {str(e)}\n"
            with open(stderr_file, 'a') as err:
                err.write(error_msg)
            
            # Combine output including the error message
            output = f"STDOUT:\n{stdout_content}\n\nSTDERR:\n{stderr_content}{error_msg}"
            return False, output
    
    def generate_final_report(self):
        """Generate a final report summarizing the entire experiment."""
        findings = self.read_file_safe(self.findings_file)
        plan = self.read_file_safe(self.plan_file)
        idea = self.read_file_safe(self.idea_file)
        
        messages = [
            Message("system", "You are writing a report on an automated ML experiment. Be thorough but concise."),
            Message("user", f"""Generate a final report for this ML experiment.

Original Idea:
{idea}

Plan:
{plan}

All Findings:
{findings}

Create a well-structured report that includes:
1. Executive Summary
2. Methodology
3. Key Results and Findings  
4. Challenges and Solutions
5. Conclusions
6. Recommendations for Future Work""")
        ]
        
        # Create context for this call
        context = CallContext(
            function_name="generate_final_report",
            experiment_id=self.experiment_id
        )
        
        response = self.client.chat_completion(
            messages=messages,
            model=self.model_config.report_model,
            temperature=0.5,
            max_tokens=self.config.max_tokens_report,
            context=context
        )
        
        report = self.client.get_completion_text(response)
        report_file = self.experiment_dir / "REPORT.md"
        report_file.write_text(report)
        
        return report
    
    def initialize_experiment(self):
        """Initialize the experiment with IDEA.md and generate initial PLAN.md and TASKS.md."""
        if not self.idea_file.exists():
            log_error("Error: IDEA.md not found. Please create it first.")
            sys.exit(1)
        
        idea = self.idea_file.read_text()
        
        # Get environment information
        env_info = get_environment_info()
        env_summary = f"""
Environment Information:
- Python: {env_info['python_version'].split()[0]}
- Platform: {env_info['platform']}
- PyTorch: {env_info['torch_version'] or 'Not installed'}
- GPU: {'Available' if env_info['gpu_available'] else 'Not available'}
"""
        if env_info['gpu_info']:
            gpu = env_info['gpu_info']
            env_summary += f"  - {gpu['name']} ({gpu['memory_gb']:.1f}GB, Compute {gpu['compute_capability']})\n"
        
        if env_info['installed_packages']:
            env_summary += "\nKey Installed Packages:\n"
            for pkg in sorted(env_info['installed_packages'])[:20]:  # Show top 20 packages
                env_summary += f"  - {pkg}\n"
        
        # Create plan generation messages
        messages = [
            Message("system", system_prompt("""an expert ML researcher creating experimental plans. 
You should create a detailed, actionable plan based on the research idea provided.""")),
            Message("user", f"""Create a detailed experimental plan based on the following idea:
                    
IDEA.md:
```
{idea}
```

Environment Information:
```
{env_summary}
```

The plan should include:
1. Clear objectives and success criteria
2. Methodology and approach
3. Implementation milestones
4. Resource requirements
5. Expected challenges and mitigation strategies
6. Evaluation metrics

Important:
- Always use fp32 by default to avoid NaN issues
- If GPU is not available, ensure the plan accounts for CPU-only execution.
- Include all relevant information from the idea document
- We should avoid creating files unless absolutely necessary. Within the automated environment results are passed through 
  standard output, datasets should be processed via streaming.

Format as a structured markdown document.""")
        ]
        
        # Create context for this call
        context = CallContext(
            function_name="initialize_experiment",
            experiment_id=self.experiment_id,
            additional_context={"phase": "plan_generation"}
        )
        
        response = self.client.chat_completion(
            messages=messages,
            model=self.model_config.plan_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens_code,
            context=context
        )
        
        plan = self.client.get_completion_text(response)
        self.plan_file.write_text(plan)
        log_success("Generated PLAN.md")
        
        # Generate initial tasks if none exist
        if not self.task_manager.tasks:
            log_progress("Generating initial task list")
            
            plan = self.plan_file.read_text()
            
            messages = [
                Message("system", system_prompt("""a task manager for ML experiments. 
Create a comprehensive task list based on the experimental plan.""")),
                Message("user", f"""Create a task list for this ML experiment.

Experimental Idea:
{idea}

Experimental Plan:
{plan}

Generate a list of specific, actionable tasks that will help achieve the experimental goals.
Each task should be:
- Clear and specific
- Achievable within a single run
- Ordered by logical dependencies
- Marked with appropriate priority (high, medium, low)

Respond with a JSON object containing 'tasks' array, where each task has:
- description: Clear description of what needs to be done
- priority: 'high', 'medium', or 'low'
- dependencies: List of task descriptions this depends on (optional)""")
            ]
            
            # Create context for this call
            context = CallContext(
                function_name="initialize_experiment",
                experiment_id=self.experiment_id,
                additional_context={"phase": "task_generation"}
            )
            
            response = self.client.chat_completion(
                messages=messages,
                model=self.model_config.analysis_model,
                temperature=self.config.temperature,
                response_format={"type": "json_object", "json_schema": {"name": "task_updates", "schema": INITIAL_TASKS_SCHEMA}},
                context=context
            )
            
            result = json.loads(self.client.get_completion_text(response))
            
            # Add tasks
            for task_data in result.get('tasks', []):
                self.task_manager.add_task(
                    description=task_data['description'],
                    priority=task_data.get('priority', 'medium'),
                    dependencies=task_data.get('dependencies')
                )
            
            log_success(f"Generated {len(self.task_manager.tasks)} initial tasks")
        
        # Initialize findings file if it doesn't exist
        if not self.findings_file.exists():
            initial_content = f"""# Experiment Log

**Experiment ID:** {self.experiment_id}
**Started:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
This log tracks the progress, findings, and insights from each experimental run.

---
"""
            self.findings_file.write_text(initial_content)
    
    def run_experiment(self):
        """Run the main experiment loop."""
        # Initialize experiment files
        self.initialize_experiment()
        
        # Get starting run number
        current_run = self.get_current_run_number() + 1
        
        log_panel(f"Starting ML Experiment Runner\nBeginning from run {current_run}", "Auto ML Runner")
        
        while current_run <= self.config.max_runs:
            log_section("Run", current_run, self.config.max_runs)
            
            # Create run directory
            run_dir = self.create_run_directory(current_run)
            
            # Get previous code and logs
            previous_code = self.get_previous_code(current_run)
            previous_logs = self.get_logs(current_run - 1)
            
            # Summarize previous logs if available
            log_summary = None
            if previous_logs:
                with Spinner("Summarizing previous run..."):
                    log_summary = self.get_or_create_log_summary(current_run - 1, previous_logs)
                log_success("Summarized previous run")
            
            # Generate code for this run
            with Spinner("Generating code..."):
                working_tasks, code = self.generate_code(current_run, previous_code, log_summary)
            
            log_success("Generated new code")
            
            # Mark tasks as in_progress
            if working_tasks:
                for task_id in working_tasks:
                    self.task_manager.update_task(task_id, status='in_progress')
                log_info(f"Working on tasks: {', '.join(working_tasks)}")
            
            # Save generated code
            code_file = run_dir / "main.py"
            code_file.write_text(code)
            
            # Run training
            success, output = self.run_training(run_dir, code)
            
            if success:
                log_success("Training completed successfully")
            else:
                log_error("Training failed")
            
            # Summarize this run's results
            with Spinner("Analyzing results..."):
                current_summary = self.get_or_create_log_summary(current_run, output)
            
            # Analyze run results and update tasks
            analysis_result = self.analyze_run_results(current_run, current_summary, code, success)
            log_success("Analyzed run results")
            
            experiment_state = analysis_result.get('experiment_state', {
                'experiment_complete': False,
                'plan_revision_needed': False,
                'early_exit_required': False
            })
            
            if experiment_state.get('early_exit_required', False):
                reason = experiment_state.get('reason', 'Cannot make further progress')
                log_warning(f"Early exit required: {reason}", prefix="⚠️")
                break


            if experiment_state.get('plan_revision_needed', False):
                reason = analysis_result.get('reason', 'Plan needs adjustment based on findings')
                log_info(f"Revising experimental plan: {reason}")
                self.revise_plan(reason)

            # Update tasks based on analysis
            self.update_tasks_from_analysis(current_run, analysis_result, current_summary, self.read_file_safe(self.plan_file), code)
            log_success("Updated task list")

            # Show progress summary
            task_summary = self.task_manager.get_structured_state()
            progress_msg = f"Progress: Run {current_run}/{self.config.max_runs} | "
            progress_msg += f"Tasks: {task_summary['completed']}/{task_summary['total_tasks']} completed"
            if task_summary['in_progress'] > 0:
                progress_msg += f" ({task_summary['in_progress']} in progress)"
            log_status(progress_msg, "progress")
            
            # Handle experiment state flags
            if experiment_state.get('experiment_complete', False):
                reason = experiment_state.get('reason', 'Experimental goals achieved')
                log_success(f"Experiment complete: {reason}", prefix="🎯")
                break            
            
            current_run += 1
        
        # Generate final report
        with Spinner("Generating final report..."):
            report = self.generate_final_report()
        log_success("Generated REPORT.md")
        
        # Display summary
        log_panel(Markdown(report[:1000] + "...\n\n[See REPORT.md for full report]"), "Experiment Summary")


@click.command()
@click.option('--experiment-dir', '-d', type=click.Path(path_type=Path), 
              default=Path.cwd(), help='Experiment directory')
@click.option('--max-runs', '-r', type=int, help='Maximum number of runs')
@click.option('--resume', is_flag=True, help='Resume from last run')
def main(experiment_dir: Path, max_runs: Optional[int], resume: bool):
    """Auto ML Runner - Automated ML experiment runner using LLMs."""
    # Load configurations
    model_config = ModelConfig()
    run_config = RunConfig()
    
    if max_runs:
        run_config.max_runs = max_runs
    
    # Create runner
    runner = ExperimentRunner(experiment_dir, run_config, model_config)
    
    try:
        runner.run_experiment()
    except KeyboardInterrupt:
        log_warning("\nExperiment interrupted by user")
    except Exception as e:
        log_error(f"\nError: {str(e)}")
        raise


if __name__ == "__main__":
    main()