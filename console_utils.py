"""Console utilities for semantic output styling."""
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from typing import Optional, Union

console = Console()


def log_info(message: str, prefix: Optional[str] = None):
    """Log an informational message."""
    if prefix:
        console.print(f"{prefix} {message}")
    else:
        console.print(message)


def log_success(message: str, prefix: str = "✓"):
    """Log a success message."""
    text = Text()
    text.append(f"{prefix} ", style="success")
    text.append(message)
    console.print(text)


def log_error(message: str, prefix: str = "✗"):
    """Log an error message."""
    text = Text()
    text.append(f"{prefix} ", style="error")
    text.append(message)
    console.print(text)


def log_warning(message: str, prefix: str = "⚠"):
    """Log a warning message."""
    text = Text()
    text.append(f"{prefix} ", style="warning")
    text.append(message)
    console.print(text)


def log_progress(message: str):
    """Log a progress/working message."""
    console.print(f"[dim]{message}...[/dim]")


def log_section(title: str, run_number: Optional[int] = None, total_runs: Optional[int] = None):
    """Log a section header."""
    if run_number and total_runs:
        header = f"═══ Run {run_number}/{total_runs} ═══"
    else:
        header = f"═══ {title} ═══"
    
    text = Text(header, style="bold")
    console.print(f"\n{text}")


def log_panel(content: Union[str, object], title: str):
    """Display content in a panel."""
    console.print(Panel(content, title=title))


def log_status(message: str, status: str):
    """Log a message with a specific status type."""
    status_map = {
        "success": log_success,
        "error": log_error,
        "warning": log_warning,
        "info": log_info,
        "progress": log_progress
    }
    
    func = status_map.get(status.lower(), log_info)
    func(message)