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
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.markdown import Markdown

from llm_client import OpenRouterClient, Message
from config import ModelConfig, RunConfig
from console_utils import (
    log_info, log_success, log_error, log_warning, 
    log_progress, log_section, log_panel, log_status
)
from task_manager import TaskManager
from schemas import LOG_SUMMARY_SCHEMA, ANALYSIS_AND_TASKS_SCHEMA, INITIAL_TASKS_SCHEMA

console = Console()

SYSTEM_PROMPT_CONTEXT = "We are conducting automated ML experiments.\n "

def system_prompt(role: str) -> str:
    return f"""
{SYSTEM_PROMPT_CONTEXT}
Your role is {role}.
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
            # Filter to common ML/data science packages
            ml_packages = {
                'torch', 'tensorflow', 'jax', 'numpy', 'pandas', 'scikit-learn',
                'matplotlib', 'seaborn', 'plotly', 'transformers', 'datasets',
                'accelerate', 'bitsandbytes', 'peft', 'tokenizers', 'scipy',
                'opencv-python', 'pillow', 'tqdm', 'wandb', 'tensorboard',
                'lightning', 'pytorch-lightning', 'torchvision', 'torchaudio'
            }
            env_info["installed_packages"] = [
                f"{pkg['name']}=={pkg['version']}" 
                for pkg in packages 
                if pkg['name'].lower() in ml_packages or 
                   any(keyword in pkg['name'].lower() for keyword in ['torch', 'tensor', 'cuda'])
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
        self.client = OpenRouterClient()
        
        # Create experiment directory if it doesn't exist
        self.experiment_dir.mkdir(exist_ok=True)
        
        # Create shared files directory for persistent data across runs
        self.shared_files_dir = self.experiment_dir / "shared_files"
        self.shared_files_dir.mkdir(exist_ok=True)
        
        # Initialize state files
        self.plan_file = self.experiment_dir / "PLAN.md"
        self.tasks_file = self.experiment_dir / "tasks.json"
        self.findings_file = self.experiment_dir / "KEY_FINDINGS.md"
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
    
    def get_previous_logs(self, run_number: int) -> Optional[str]:
        """Get the logs from the previous run."""
        if run_number <= 1:
            return None
        
        prev_run_dir = self.experiment_dir / f"run_{run_number - 1}"
        prev_log_file = prev_run_dir / "output.log"
        
        if prev_log_file.exists():
            return prev_log_file.read_text()
        return None
    
    def summarize_logs(self, logs: str, run_number: int) -> str:
        """Use LLM to create a mechanical summary of the logs."""
        # First extract key information
        key_info = extract_key_log_info(logs)
        
        # Build a structured summary of extracted info
        extracted_summary = ""
        if key_info['errors']:
            extracted_summary += "ERRORS FOUND:\n"
            for error in key_info['errors'][:3]:  # Limit to first 3 errors
                extracted_summary += f"{error}\n\n"
        
        if key_info['final_metrics']:
            extracted_summary += "FINAL METRICS:\n"
            for metric in key_info['final_metrics'][-5:]:  # Last 5 metrics
                extracted_summary += f"{metric}\n"
            extracted_summary += "\n"
        
        if key_info['gpu_memory']:
            extracted_summary += "GPU MEMORY:\n"
            for mem in key_info['gpu_memory'][-3:]:  # Last 3 GPU memory reports
                extracted_summary += f"{mem}\n"
            extracted_summary += "\n"
        
        # Then truncate the full logs
        rlogs = restrict_text(logs, 50000, 1000, remove_middle=True)
        
        messages = [
            Message("system", system_prompt("""to extract and copy important information from training logs.
Do NOT interpret or analyze - just extract factual information.
Focus on: final metrics, errors, warnings, GPU usage, and important output lines.""")),
            Message("user", f"""Extract key information from these training logs:

{extracted_summary}

FULL LOGS (truncated):
------
{rlogs}
------

Extract and copy important information from training logs.
Do NOT interpret or analyze - just extract factual information.""")
        ]
        
        response = self.client.chat_completion(
            messages=messages,
            model=self.model_config.summarize_model,
            temperature=0.1,  # Low temperature for factual extraction
            max_tokens=self.config.max_tokens_summary
        )
        
        return self.client.get_completion_text(response)
    
    def generate_code(self, run_number: int, previous_code: Optional[str], log_summary: Optional[str]) -> str:
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
- Always use fp32 by default to avoid NaN issues
- Include proper error handling and logging
- Log output should go to standard out and standard error, which will be captured by the experiment runner for further analysis
- Do not use tqdm for progress bars
- Do not create any files unless it's necessary according to the plan
- Make incremental improvements based on previous results
- Focus on achieving the experimental goals
- Consider the available hardware and installed packages

File Management:
- You can access files from the previous run using relative paths: ../run_{run_number - 1}/filename
- Shared persistent files (datasets, pretrained models, etc.) should be accessed from: ../shared_files/
- If your code produces files needed by future runs (e.g., model checkpoints), save them in the current directory
- If you need to share immutable reference data across all runs, copy it to ../shared_files/
- The plan will specify file paths relative to the run directory when files are needed
- Check the plan for any file paths mentioned - they indicate resources you should use

General Plan:
{plan}

Key Findings So Far:
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
        Return ONLY the Python code wrapped in ```python...```, no explanations."""
        
        messages = [
            Message("system", system_prompt_text),
            Message("user", user_prompt)
        ]
        
        response = self.client.chat_completion(
            messages=messages,
            model=self.model_config.code_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens_code
        )
        
        code = self.client.get_completion_text(response).strip()
        
        # Clean up code formatting
        if code.startswith("```python"):
            code = code[9:]
        if code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        
        return code.strip()
    
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

Key Findings So Far:
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
5. Addresses any blockers or challenges discovered
6. Updates file management strategy if needed:
   - Any new shared data/models should reference the ../shared_files/ directory
   - Specify relative paths for any files discovered during experiments
7. Considers the current environment and available packages

Format as a structured markdown document.""")
        ]
        
        response = self.client.chat_completion(
            messages=messages,
            model=self.model_config.plan_model,
            temperature=0.3,
            max_tokens=5000
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
    
    def update_global_state(self, run_number: int, log_summary: str, code: str, success: bool):
        """Analyze results and update tasks in a single LLM call."""
        # Get current state
        current_findings = self.read_file_safe(self.findings_file)
        task_state = self.task_manager.get_structured_state()
        plan = self.read_file_safe(self.plan_file)
        
        # Get code changes summary
        previous_code = self.get_previous_code(run_number)
        code_changes = get_code_changes(previous_code, code)
        
        # Create prompt for combined analysis
        messages = [
            Message("system", system_prompt("""an expert ML engineer analyzing experiment results. 
Provide a structured JSON response with:
1. 'analysis': Your interpretation of the run results
2. 'key_findings': Key discoveries from this run
3. 'experiment_state': High-level experiment status flags
4. 'task_updates': List of task updates based on the results""")),

            Message("user", f"""Analyze the results from run {run_number} and update the experiment state.

Code Changes Summary:
{code_changes}

Code from Run {run_number}:
------
{code}
------

Log Summary from Run {run_number}:
------
{log_summary}
------

Current Tasks (Total: {task_state['total_tasks']}, Pending: {task_state['pending']}, In Progress: {task_state['in_progress']}, Completed: {task_state['completed']}):
{json.dumps(task_state['tasks'], indent=2)}

Recent Findings:
------
{restrict_text(current_findings, 10000, 1000, remove_middle=False) if current_findings else 'No previous findings'}
------

Analyze what happened and provide:
1. A detailed analysis of the results
2. Key findings (max 5 bullet points)
3. Experiment state flags:
   - experiment_complete: Have we achieved the experimental goals?
   - plan_revision_needed: Should we revise the experimental plan based on current progress?
   - early_exit_required: Are there blockers preventing further progress?
4. Task updates based on the results

For task updates, you can:
- Mark tasks as complete (with notes)
- Add new tasks discovered during the run
- Update task status to blocked if there are dependencies""")
        ]
        
        response = self.client.chat_completion(
            messages=messages,
            model=self.model_config.code_model,
            temperature=0.3,
            max_tokens=3000,
            response_format={"type": "json_schema", "json_schema": {"name": "analysis_tasks", "schema": ANALYSIS_AND_TASKS_SCHEMA}}
        )
        
        try:
            result = json.loads(self.client.get_completion_text(response))
            
            # Update KEY_FINDINGS.md with the analysis and key findings
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            finding = f"\n\n## Run {run_number} - {timestamp}\n\n### Analysis\n{result['analysis']}\n\n### Key Findings\n"
            for kf in result.get('key_findings', []):
                finding += f"- {kf}\n"
            
            with open(self.findings_file, "a") as f:
                f.write(finding)
            
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
            
            # Return experiment state flags
            return result.get('experiment_state', {
                'experiment_complete': False,
                'plan_revision_needed': False,
                'early_exit_required': False
            })
            
        except Exception as e:
            log_error(f"Failed to parse LLM response: {e}")
            # Fallback: just save the summary as finding
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            finding = f"\n\n## Run {run_number} - {timestamp}\n\n### Log Summary\n{json.dumps(log_summary, indent=2)}\n"
            with open(self.findings_file, "a") as f:
                f.write(finding)
            
            # Return default state
            return {
                'experiment_complete': False,
                'plan_revision_needed': False,
                'early_exit_required': False
            }
    
    
    def run_training(self, run_dir: Path, code: str) -> Tuple[bool, str]:
        """Execute the training code and capture output."""
        code_file = run_dir / "main.py"
        log_file = run_dir / "output.log"
        
        # Write code to file
        code_file.write_text(code)
        
        # Run the training script
        log_progress(f"Running training script in {run_dir}")
        
        try:
            result = subprocess.run(
                [sys.executable, "main.py"],
                cwd=run_dir,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds
            )
            
            # Combine stdout and stderr
            output = f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
            log_file.write_text(output)
            
            success = result.returncode == 0
            return success, output
            
        except subprocess.TimeoutExpired:
            error_msg = f"Training script timed out after {self.config.timeout_seconds} seconds"
            log_file.write_text(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Error running training script: {str(e)}"
            log_file.write_text(error_msg)
            return False, error_msg
    
    def generate_final_report(self):
        """Generate a final report summarizing the entire experiment."""
        findings = self.read_file_safe(self.findings_file)
        plan = self.read_file_safe(self.plan_file)
        idea = self.read_file_safe(self.idea_file)
        
        messages = [
            Message("system", "You are writing a comprehensive report on an ML experiment. Be thorough but concise."),
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
        
        response = self.client.chat_completion(
            messages=messages,
            model=self.model_config.report_model,
            temperature=0.5,
            max_tokens=self.config.max_tokens_report
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
        
        # Generate initial plan if it doesn't exist
        if not self.plan_file.exists():
            log_progress("Generating initial experiment plan")
            
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
            
            messages = [
                Message("system", system_prompt("""an expert ML researcher creating detailed experimental plans""")),
                Message("user", f"""Based on this idea, create a detailed experimental plan:

------
{idea}
------

{env_summary}

The plan should include:
1. Clear objectives and success criteria
2. Technical approach and methodology
3. Key milestones and checkpoints
4. Potential challenges and mitigation strategies
5. Expected outcomes
6. File management strategy (if applicable):
   - Any shared data/models should reference the ../shared_files/ directory
   - Specify relative paths for any files mentioned in the idea
   - Note which files need to persist across runs
   - Note that some frameworks might cache files transparently, that makes things easier

Important: 
- If the idea mentions any data files, pretrained models, or other resources,
  specify their paths relative to the run directory (e.g., ../shared_files/dataset.csv).
- Consider the available compute resources and installed packages when designing the approach.
- If GPU is not available, ensure the plan accounts for CPU-only execution.

Format as a structured markdown document.""")
            ]
            
            response = self.client.chat_completion(
                messages=messages,
                model=self.model_config.plan_model,
                temperature=0.1,
                max_tokens=5000
            )
            
            plan = self.client.get_completion_text(response)
            self.plan_file.write_text(plan)
            log_success("Generated PLAN.md")
        
        # Generate initial tasks if it doesn't exist
        if not self.tasks_file.exists():
            log_progress("Generating initial task list")
            
            plan = self.plan_file.read_text()
            messages = [
                Message("system", system_prompt("""a project manager for ML experiments. Create a comprehensive task list.""")),
                Message("user", f"""Based on this experimental plan, create a task list:

{plan}

Note: We generally assume that the environment is set up, but it's worth checking that it functions as expected.
This which are already described in the plan do not require detailed descriptions.
Organize by priority and include dependencies where relevant.""")
            ]
            
            response = self.client.chat_completion(
                messages=messages,
                model=self.model_config.tasks_model,
                temperature=0.3,
                max_tokens=2000,
                response_format={"type": "json_schema", "json_schema": {"name": "initial_tasks", "schema": INITIAL_TASKS_SCHEMA}}
            )
            
            try:
                # Parse response and create tasks
                result = json.loads(self.client.get_completion_text(response))
                task_list = result.get('tasks', [])
                
                for task_data in task_list:
                    self.task_manager.add_task(
                        description=task_data['description'],
                        priority=task_data.get('priority', 'medium')
                    )
                
                log_success(f"Generated {len(task_list)} initial tasks")
            except Exception as e:
                log_error(f"Failed to generate initial tasks: {e}")
                # Create a default task
                self.task_manager.add_task(
                    "Implement initial training script",
                    priority="high"
                )
        
        # Initialize KEY_FINDINGS.md if it doesn't exist
        if not self.findings_file.exists():
            self.findings_file.write_text("# Key Findings\n\nThis document tracks important discoveries and results from each run.\n")
            log_success("Initialized KEY_FINDINGS.md")
    
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
            previous_logs = self.get_previous_logs(current_run)
            
            # Summarize previous logs if available
            log_summary = None
            if previous_logs:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task("Summarizing previous run...", total=None)
                    log_summary = self.summarize_logs(previous_logs, current_run - 1)
                    progress.remove_task(task)
                
                log_success("Summarized previous run")
            
            # Generate code for this run
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Generating code...", total=None)
                code = self.generate_code(current_run, previous_code, log_summary)
                progress.remove_task(task)
            
            log_success("Generated new code")
            
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
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Analyzing results...", total=None)
                current_summary = self.summarize_logs(output, current_run)
                progress.remove_task(task)
            
            # Update global state
            experiment_state = self.update_global_state(current_run, current_summary, code, success)
            log_success("Updated experiment state")
            
            # Detect common issues
            detected_issues = self.detect_common_issues(current_run, output)
            if detected_issues:
                log_warning(f"Potential issues detected:\n{detected_issues}", prefix="⚠️")
            
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
            
            if experiment_state.get('early_exit_required', False):
                reason = experiment_state.get('reason', 'Cannot make further progress')
                log_warning(f"Early exit required: {reason}", prefix="⚠️")
                break
            
            if experiment_state.get('plan_revision_needed', False):
                reason = experiment_state.get('reason', 'Plan needs adjustment based on findings')
                log_info(f"Revising experimental plan: {reason}")
                self.revise_plan(reason)
            
            current_run += 1
        
        # Generate final report
        log_progress("\nGenerating final report")
        report = self.generate_final_report()
        log_success("Generated REPORT.md")
        
        # Display summary
        log_panel(Markdown(report[:1000] + "...\n\n[See REPORT.md for full report]"), "Experiment Summary")

    def detect_common_issues(self, run_number: int, current_summary: str) -> Optional[str]:
        """Detect common ML experiment issues that might need intervention."""
        issues = []
        
        # Check for repeated errors across runs
        if run_number > 2:
            recent_summaries = []
            for i in range(max(1, run_number - 3), run_number):
                prev_run_dir = self.experiment_dir / f"run_{i}"
                prev_log = prev_run_dir / "output.log"
                if prev_log.exists():
                    # Extract key info from previous logs
                    prev_info = extract_key_log_info(prev_log.read_text())
                    recent_summaries.append(prev_info)
            
            # Check for recurring errors
            current_errors = extract_key_log_info(current_summary).get('errors', [])
            if current_errors:
                error_pattern = current_errors[0].lower() if current_errors else ""
                similar_error_count = sum(
                    1 for summary in recent_summaries 
                    if any(error_pattern in str(e).lower() for e in summary.get('errors', []))
                )
                if similar_error_count >= 2:
                    issues.append(f"Recurring error pattern detected across {similar_error_count + 1} runs")
            
            # Check for OOM patterns
            oom_keywords = ['out of memory', 'oom', 'cuda out of memory', 'allocate']
            current_has_oom = any(keyword in current_summary.lower() for keyword in oom_keywords)
            prev_oom_count = sum(
                1 for summary in recent_summaries
                if any(keyword in str(summary).lower() for keyword in oom_keywords)
            )
            if current_has_oom and prev_oom_count >= 1:
                issues.append("Repeated out-of-memory errors - consider reducing batch size or model size")
        
        # Check for no progress (metrics not improving)
        if run_number > 3:
            # This is a simple check - could be made more sophisticated
            if 'loss' in current_summary.lower():
                # Look for patterns like "loss: X.XX" or "loss = X.XX"
                import re
                loss_pattern = r'loss[:\s=]+(\d+\.?\d*)'
                current_losses = re.findall(loss_pattern, current_summary.lower())
                if current_losses and len(current_losses) > 0:
                    # Check if loss is very high (potential divergence)
                    try:
                        last_loss = float(current_losses[-1])
                        if last_loss > 100:
                            issues.append(f"Loss appears to be diverging ({last_loss:.2f}) - check learning rate")
                    except:
                        pass
        
        return "\n".join(issues) if issues else None


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