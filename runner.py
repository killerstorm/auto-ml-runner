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

main_py = "main.py"

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
            lines = lines[:max_lines//2] + ["..."] + lines[-max_lines//2:]
        else:
            lines = ["..."] + lines[-max_lines:]
    
    # Rejoin lines
    text = "\n".join(lines)
    
    # Handle character length restriction
    if len(text) > max_length:
        text = "..." + text[-(max_length-3):]
            
    return text



class ExperimentRunner:
    def __init__(self, experiment_dir: Path, config: RunConfig, model_config: ModelConfig):
        self.experiment_dir = experiment_dir
        self.config = config
        self.model_config = model_config
        self.client = OpenRouterClient()
        
        # Create experiment directory if it doesn't exist
        self.experiment_dir.mkdir(exist_ok=True)
        
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
        rlogs = restrict_text(logs, 50000, 1000, remove_middle=True)
        messages = [
            Message("system", """You are a log analyzer. Extract and copy important information from training logs.
Do NOT interpret or analyze - just extract factual information.
Focus on: final metrics, errors, warnings, GPU usage, and important output lines."""),
            Message("user", f"""Extract key information from these training logs:

------
{rlogs}
------

PExtract and copy important information from training logs.
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
        
        system_prompt = """You are an expert ML engineer implementing experiments. Generate complete, runnable Python code based on the provided context.

Important guidelines:
- Generate a complete {main_py} file, not patches or fragments
- Always use fp32 by default to avoid NaN issues
- Include proper error handling and logging
- Make incremental improvements based on previous results
- Focus on achieving the experimental goals
- Write clean, well-structured code"""
        
        user_prompt = f"""Generate the complete {main_py} code for run {run_number}.

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
            Message("system", system_prompt),
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
    
    def update_global_state(self, run_number: int, log_summary: str, success: bool):
        """Analyze results and update tasks in a single LLM call."""
        # Get current state
        current_findings = self.read_file_safe(self.findings_file)
        task_state = self.task_manager.get_structured_state()
        
        # Create prompt for combined analysis
        messages = [
            Message("system", """You are an expert ML engineer analyzing experiment results. 
Provide a structured JSON response with two sections:
1. 'analysis': Your interpretation of the run results and key findings
2. 'task_updates': List of task updates based on the results"""),
            Message("user", f"""Analyze the results from run {run_number} and update the task list.

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
3. Task updates based on the results

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
            
        except Exception as e:
            log_error(f"Failed to parse LLM response: {e}")
            # Fallback: just save the summary as finding
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            finding = f"\n\n## Run {run_number} - {timestamp}\n\n### Log Summary\n{json.dumps(log_summary, indent=2)}\n"
            with open(self.findings_file, "a") as f:
                f.write(finding)
    
    
    def run_training(self, run_dir: Path, code: str) -> Tuple[bool, str]:
        """Execute the training code and capture output."""
        code_file = run_dir / "train.py"
        log_file = run_dir / "output.log"
        
        # Write code to file
        code_file.write_text(code)
        
        # Run the training script
        log_progress(f"Running training script in {run_dir}")
        
        try:
            result = subprocess.run(
                [sys.executable, "train.py"],
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
            
            messages = [
                Message("system", "You are an expert ML researcher. Create a detailed experimental plan."),
                Message("user", f"""Based on this idea, create a detailed experimental plan:

{idea}

The plan should include:
1. Clear objectives and success criteria
2. Technical approach and methodology
3. Key milestones and checkpoints
4. Potential challenges and mitigation strategies
5. Expected outcomes

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
                Message("system", "You are a project manager for ML experiments. Create a comprehensive task list."),
                Message("user", f"""Based on this experimental plan, create a detailed task list:

{plan}

Create a comprehensive list of tasks needed to achieve the experimental goals.
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
            code_file = run_dir / "train.py"
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
            self.update_global_state(current_run, current_summary, success)
            log_success("Updated experiment state")
            
            # Check if we should stop early (e.g., if goal achieved)
            if success and "goal achieved" in current_summary.lower():
                log_success("Experimental goal achieved! Stopping early.", prefix="🎯")
                break
            
            current_run += 1
        
        # Generate final report
        log_progress("\nGenerating final report")
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