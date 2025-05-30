#!/usr/bin/env python3
"""View tasks in a formatted way."""
import sys
from pathlib import Path
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from task_manager import TaskManager

console = Console()


@click.command()
@click.option('--experiment-dir', '-d', type=click.Path(path_type=Path), 
              default=Path.cwd(), help='Experiment directory')
@click.option('--format', '-f', type=click.Choice(['table', 'markdown']), 
              default='table', help='Output format')
def main(experiment_dir: Path, format: str):
    """View tasks for an experiment."""
    tasks_file = experiment_dir / "tasks.json"
    
    if not tasks_file.exists():
        console.print("[error]No tasks.json found in experiment directory[/error]")
        sys.exit(1)
    
    task_manager = TaskManager(tasks_file)
    
    if format == 'markdown':
        console.print(task_manager.to_markdown())
    else:
        # Create a table view
        state = task_manager.get_structured_state()
        
        # Summary panel
        summary = f"""Total Tasks: {state['total_tasks']}
✅ Completed: {state['completed']}
🔄 In Progress: {state['in_progress']}
📋 Pending: {state['pending']}
🚫 Blocked: {state['blocked']}"""
        
        console.print(Panel(summary, title="Task Summary"))
        
        # Task table
        table = Table(title="Task Details")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Status", style="magenta")
        table.add_column("Priority")
        table.add_column("Description", style="white")
        table.add_column("Notes", style="dim")
        
        # Color code by priority
        priority_colors = {
            'high': 'red',
            'medium': 'yellow',
            'low': 'green'
        }
        
        for task in task_manager.tasks:
            priority_style = priority_colors.get(task.priority, 'white')
            table.add_row(
                task.id,
                task.status,
                f"[{priority_style}]{task.priority}[/{priority_style}]",
                task.description,
                task.notes or ""
            )
        
        console.print(table)


if __name__ == "__main__":
    main()