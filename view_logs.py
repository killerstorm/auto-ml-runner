#!/usr/bin/env python3
"""Interactive LLM Log Viewer

Browse and analyze LLM interaction logs with context about function calls and runs.
"""
import json
import click
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.syntax import Syntax
from rich.layout import Layout
from rich.live import Live
from rich.prompt import Prompt, Confirm
from rich.columns import Columns
from rich.tree import Tree
from collections import defaultdict
import readchar
import time

console = Console()


class LogEntry:
    """Represents a single LLM log entry."""
    
    def __init__(self, data: Dict, file_path: Path):
        self.data = data
        self.file_path = file_path
        self.request_id = data.get("request_id", "unknown")
        self.timestamp = datetime.fromisoformat(data.get("timestamp", ""))
        self.context = data.get("context", {})
        self.request = data.get("request", {})
        self.response = data.get("response", {})
        self.error = data.get("error")
        self.performance = data.get("performance", {})
        
    @property
    def function_name(self) -> str:
        return self.context.get("function_name", "unknown")
    
    @property
    def run_number(self) -> Optional[int]:
        return self.context.get("run_number")
    
    @property
    def experiment_id(self) -> Optional[str]:
        return self.data.get("experiment_id")
    
    @property
    def model(self) -> str:
        return self.request.get("model", self.request.get("payload", {}).get("model", "unknown"))
    
    @property
    def duration_ms(self) -> Optional[int]:
        return self.performance.get("duration_ms")
    
    @property
    def tokens_used(self) -> Optional[Dict]:
        return self.performance.get("tokens_used")
    
    @property
    def success(self) -> bool:
        return self.error is None and self.response is not None


class LogViewer:
    """Interactive log viewer."""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.logs: List[LogEntry] = []
        self.filtered_logs: List[LogEntry] = []
        self.current_index = 0
        self.filter_function = None
        self.filter_run = None
        self.filter_experiment = None
        self.sort_by = "timestamp"
        self.sort_reverse = False
        
    def load_logs(self):
        """Load all log files."""
        console.print("[blue]Loading logs...[/blue]")
        
        # Load individual JSON files
        json_files = list(self.log_dir.glob("*.json"))
        for file_path in json_files:
            try:
                with open(file_path) as f:
                    data = json.load(f)
                    self.logs.append(LogEntry(data, file_path))
            except Exception as e:
                console.print(f"[red]Error loading {file_path}: {e}[/red]")
        
        # Load session JSONL files
        jsonl_files = list(self.log_dir.glob("session_*.jsonl"))
        for file_path in jsonl_files:
            try:
                with open(file_path) as f:
                    for line_num, line in enumerate(f, 1):
                        if line.strip():
                            try:
                                data = json.loads(line)
                                self.logs.append(LogEntry(data, file_path))
                            except Exception as e:
                                console.print(f"[red]Error in {file_path} line {line_num}: {e}[/red]")
            except Exception as e:
                console.print(f"[red]Error loading {file_path}: {e}[/red]")
        
        # Sort by timestamp by default
        self.logs.sort(key=lambda x: x.timestamp)
        self.filtered_logs = self.logs.copy()
        
        console.print(f"[green]Loaded {len(self.logs)} log entries[/green]")
    
    def apply_filters(self):
        """Apply current filters to logs."""
        filtered = self.logs.copy()
        
        if self.filter_function:
            filtered = [log for log in filtered if log.function_name == self.filter_function]
        
        if self.filter_run is not None:
            filtered = [log for log in filtered if log.run_number == self.filter_run]
        
        if self.filter_experiment:
            filtered = [log for log in filtered if log.experiment_id == self.filter_experiment]
        
        # Apply sorting
        if self.sort_by == "timestamp":
            filtered.sort(key=lambda x: x.timestamp, reverse=self.sort_reverse)
        elif self.sort_by == "function":
            filtered.sort(key=lambda x: x.function_name, reverse=self.sort_reverse)
        elif self.sort_by == "run":
            filtered.sort(key=lambda x: (x.run_number or -1), reverse=self.sort_reverse)
        elif self.sort_by == "duration":
            filtered.sort(key=lambda x: (x.duration_ms or 0), reverse=self.sort_reverse)
        
        self.filtered_logs = filtered
        self.current_index = 0
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics for current filtered logs."""
        stats = {
            "total": len(self.filtered_logs),
            "success": sum(1 for log in self.filtered_logs if log.success),
            "errors": sum(1 for log in self.filtered_logs if not log.success),
            "by_function": defaultdict(int),
            "by_run": defaultdict(int),
            "by_model": defaultdict(int),
            "total_tokens": 0,
            "total_duration_ms": 0
        }
        
        for log in self.filtered_logs:
            stats["by_function"][log.function_name] += 1
            if log.run_number is not None:
                stats["by_run"][log.run_number] += 1
            stats["by_model"][log.model] += 1
            
            if log.tokens_used:
                stats["total_tokens"] += log.tokens_used.get("total_tokens", 0)
            if log.duration_ms:
                stats["total_duration_ms"] += log.duration_ms
        
        return stats
    
    def render_summary(self) -> Panel:
        """Render summary statistics."""
        stats = self.get_summary_stats()
        
        # Create summary table
        table1 = Table(title="Summary Statistics", show_header=False)
        table1.add_column("Metric", style="cyan")
        table1.add_column("Value", style="white")
        
        table1.add_row("Total Entries", str(stats["total"]))
        table1.add_row("Successful", f"{stats['success']} ({stats['success']/max(stats['total'], 1)*100:.1f}%)")
        table1.add_row("Errors", f"{stats['errors']} ({stats['errors']/max(stats['total'], 1)*100:.1f}%)")
        table1.add_row("Total Tokens", f"{stats['total_tokens']:,}")
        table1.add_row("Total Duration", f"{stats['total_duration_ms']/1000:.1f}s")
        
        table2 = Table(title="Summary Statistics", show_header=False)
        table2.add_column("Metric", style="cyan")
        table2.add_column("Value", style="white")

        # Add function breakdown
        if stats["by_function"]:
            table2.add_row("", "")  # Empty row
            table2.add_row("[bold]By Function[/bold]", "")
            for func, count in sorted(stats["by_function"].items(), key=lambda x: x[1], reverse=True)[:5]:
                table2.add_row(f"  {func}", str(count))
        
        # Add model breakdown
        if stats["by_model"]:
            table2.add_row("", "")  # Empty row
            table2.add_row("[bold]By Model[/bold]", "")
            for model, count in sorted(stats["by_model"].items(), key=lambda x: x[1], reverse=True):
                table2.add_row(f"  {model.split('/')[-1]}", str(count))
        
        return Panel(Columns([table1, table2]), title="📊 Statistics", border_style="blue")
    
    def render_log_entry(self, log: LogEntry) -> Panel:
        """Render a single log entry."""
        # Header with metadata
        header = Table(show_header=False, box=None)
        header.add_column("Label", style="dim")
        header.add_column("Value")
        
        header.add_row("Request ID:", log.request_id)
        header.add_row("Timestamp:", log.timestamp.strftime("%Y-%m-%d %H:%M:%S"))
        header.add_row("Function:", f"[yellow]{log.function_name}[/yellow]")
        if log.run_number is not None:
            header.add_row("Run Number:", f"[cyan]{log.run_number}[/cyan]")
        header.add_row("Model:", log.model)
        
        if log.duration_ms:
            header.add_row("Duration:", f"{log.duration_ms}ms")
        
        if log.tokens_used:
            tokens = log.tokens_used
            header.add_row("Tokens:", f"Prompt: {tokens.get('prompt_tokens', 0)}, "
                                    f"Completion: {tokens.get('completion_tokens', 0)}, "
                                    f"Total: {tokens.get('total_tokens', 0)}")
        
        # Create content sections
        content_parts = [header]
        
        # Error section if present
        if log.error:
            error_text = Text(f"\n⚠️  Error: {log.error}", style="red")
            content_parts.append(error_text)
        
        # Messages section
        messages = log.request.get("payload", {}).get("messages", [])
        if messages:
            content_parts.append(Text("\n📨 Messages:", style="bold"))
            for i, msg in enumerate(messages[-3:]):  # Show last 3 messages
                role = msg.get("role", "unknown")
                content = msg.get("content", "")[:200]  # Truncate long messages
                if len(msg.get("content", "")) > 200:
                    content += "..."
                
                role_style = {"system": "blue", "user": "green", "assistant": "yellow"}.get(role, "white")
                content_parts.append(Text(f"  [{role}]: {content}", style=role_style))
        
        # Response section
        if log.response and log.success:
            response_content = log.response.get("choices", [{}])[0].get("message", {}).get("content", "")
            if response_content:
                content_parts.append(Text("\n🤖 Response:", style="bold"))
                truncated = response_content[:500]
                if len(response_content) > 500:
                    truncated += "..."
                content_parts.append(Text(f"  {truncated}", style="dim"))
        
        # Additional context
        if log.context.get("additional"):
            content_parts.append(Text(f"\n📎 Additional Context: {log.context['additional']}", style="dim"))
        
        # Combine all parts
        content = Text()
        for part in content_parts:
            if isinstance(part, Text):
                content.append(part)
            else:
                content.append(Text.from_markup(str(part)))
        
        title = f"Log Entry {self.current_index + 1}/{len(self.filtered_logs)}"
        if not log.success:
            title += " [red]❌ Error[/red]"
        else:
            title += " [green]✓ Success[/green]"
        
        return Panel(content, title=title, border_style="green" if log.success else "red")
    
    def render_help(self) -> Panel:
        """Render help panel."""
        help_text = """
[bold]Navigation:[/bold]
  ← / → : Previous/Next entry
  Home/End : First/Last entry
  Page Up/Down : Jump 10 entries

[bold]Filtering:[/bold]
  f : Filter by function
  r : Filter by run number
  e : Filter by experiment
  c : Clear all filters

[bold]Sorting:[/bold]
  s : Change sort order
  S : Toggle sort direction

[bold]Display:[/bold]
  d : Show detailed view
  j : Export current entry as JSON
  J : Export all filtered entries as JSONL

[bold]Analysis:[/bold]
  a : Show statistics
  t : Show timeline view
  
[bold]Other:[/bold]
  h : Show this help
  q : Quit
"""
        return Panel(help_text.strip(), title="⌨️  Keyboard Shortcuts", border_style="dim")
    
    def show_timeline(self):
        """Show timeline view of logs."""
        console.clear()
        
        # Group logs by run and function
        timeline = defaultdict(list)
        for log in self.filtered_logs:
            key = (log.run_number or -1, log.function_name)
            timeline[key].append(log)
        
        # Create timeline tree
        tree = Tree("📅 Timeline View")
        
        current_run = None
        for (run, func), logs in sorted(timeline.items()):
            if run != current_run:
                if run == -1:
                    run_branch = tree.add("[dim]No Run Number[/dim]")
                else:
                    run_branch = tree.add(f"[bold cyan]Run {run}[/bold cyan]")
                current_run = run
            
            func_branch = run_branch.add(f"[yellow]{func}[/yellow] ({len(logs)} calls)")
            
            for log in logs[:5]:  # Show first 5 logs per function
                time_str = log.timestamp.strftime("%H:%M:%S")
                status = "✓" if log.success else "❌"
                duration = f" ({log.duration_ms}ms)" if log.duration_ms else ""
                func_branch.add(f"{status} {time_str} - {log.model.split('/')[-1]}{duration}")
            
            if len(logs) > 5:
                func_branch.add(f"[dim]... and {len(logs) - 5} more[/dim]")
        
        console.print(tree)
        console.print("\n[dim]Press any key to return...[/dim]")
        console.input()
    
    def export_current(self):
        """Export current log entry as JSON."""
        if not self.filtered_logs:
            console.print("[red]No logs to export[/red]")
            return
        
        log = self.filtered_logs[self.current_index]
        filename = f"export_{log.request_id}_{int(time.time())}.json"
        
        with open(filename, 'w') as f:
            json.dump(log.data, f, indent=2)
        
        console.print(f"[green]Exported to {filename}[/green]")
        time.sleep(1)
    
    def export_all(self):
        """Export all filtered logs as JSONL."""
        if not self.filtered_logs:
            console.print("[red]No logs to export[/red]")
            return
        
        filename = f"export_filtered_{int(time.time())}.jsonl"
        
        with open(filename, 'w') as f:
            for log in self.filtered_logs:
                json.dump(log.data, f)
                f.write('\n')
        
        console.print(f"[green]Exported {len(self.filtered_logs)} entries to {filename}[/green]")
        time.sleep(1)
    
    def show_detailed_view(self):
        """Show detailed view of current log entry."""
        if not self.filtered_logs:
            return
        
        log = self.filtered_logs[self.current_index]
        console.clear()
        
        # Show full request
        console.print(Panel(
            Syntax(json.dumps(log.request, indent=2), "json", theme="monokai"),
            title="📤 Full Request",
            border_style="blue"
        ))
        
        # Show full response
        if log.response:
            console.print(Panel(
                Syntax(json.dumps(log.response, indent=2), "json", theme="monokai"),
                title="📥 Full Response",
                border_style="green"
            ))
        
        # Show full context
        console.print(Panel(
            Syntax(json.dumps(log.context, indent=2), "json", theme="monokai"),
            title="🔍 Context",
            border_style="yellow"
        ))
        
        console.print("\n[dim]Press any key to return...[/dim]")
        console.input()
    
    def run(self):
        """Run the interactive viewer."""
        self.load_logs()
        
        while True:
            console.clear()
            
            # Create layout
            if self.filtered_logs:
                main_panel = self.render_log_entry(self.filtered_logs[self.current_index])
            else:
                main_panel = Panel("[red]No logs match current filters[/red]", title="No Data")
            
            summary_panel = self.render_summary()
            
            # Show current filters
            filter_text = []
            if self.filter_function:
                filter_text.append(f"Function: {self.filter_function}")
            if self.filter_run is not None:
                filter_text.append(f"Run: {self.filter_run}")
            if self.filter_experiment:
                filter_text.append(f"Experiment: {self.filter_experiment}")
            
            if filter_text:
                filter_panel = Panel(" | ".join(filter_text), title="🔍 Active Filters", border_style="yellow")
                console.print(filter_panel)
            
            # Display panels
            console.print(Columns([summary_panel, main_panel]))
            
            # Show navigation hint
            console.print(f"\n[dim]Use arrow keys to navigate, 'h' for help, 'q' to quit[/dim]")
            
            # Get user input
            key = readchar.readkey()
            
            if key.lower() == 'q':
                break
            elif key.lower() == 'h':
                console.clear()
                console.print(self.render_help())
                console.print("\n[dim]Press any key to continue...[/dim]")
                console.input()
            elif key == '\x1b[D':  # Left arrow
                if self.current_index > 0:
                    self.current_index -= 1
            elif key == '\x1b[C':  # Right arrow
                if self.current_index < len(self.filtered_logs) - 1:
                    self.current_index += 1
            elif key.lower() == 'f':
                # Filter by function
                functions = sorted(set(log.function_name for log in self.logs))
                console.print("\nAvailable functions:")
                for i, func in enumerate(functions):
                    console.print(f"  {i+1}. {func}")
                choice = Prompt.ask("Select function (number or name, empty to clear)")
                
                if choice:
                    try:
                        idx = int(choice) - 1
                        if 0 <= idx < len(functions):
                            self.filter_function = functions[idx]
                    except ValueError:
                        if choice in functions:
                            self.filter_function = choice
                else:
                    self.filter_function = None
                
                self.apply_filters()
            elif key.lower() == 'r':
                # Filter by run
                runs = sorted(set(log.run_number for log in self.logs if log.run_number is not None))
                if runs:
                    console.print(f"\nAvailable runs: {', '.join(map(str, runs))}")
                    choice = Prompt.ask("Enter run number (empty to clear)")
                    if choice:
                        try:
                            self.filter_run = int(choice)
                        except ValueError:
                            console.print("[red]Invalid run number[/red]")
                    else:
                        self.filter_run = None
                    self.apply_filters()
            elif key.lower() == 'c':
                # Clear filters
                self.filter_function = None
                self.filter_run = None
                self.filter_experiment = None
                self.apply_filters()
            elif key.lower() == 's':
                # Change sort
                options = ["timestamp", "function", "run", "duration"]
                console.print("\nSort by:")
                for i, opt in enumerate(options):
                    console.print(f"  {i+1}. {opt}")
                choice = Prompt.ask("Select sort field (1-4)")
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(options):
                        self.sort_by = options[idx]
                        self.apply_filters()
                except ValueError:
                    pass
            elif key.upper() == 'S':
                # Toggle sort direction
                self.sort_reverse = not self.sort_reverse
                self.apply_filters()
            elif key.lower() == 'd':
                self.show_detailed_view()
            elif key.lower() == 'j':
                self.export_current()
            elif key.upper() == 'J':
                self.export_all()
            elif key.lower() == 'a':
                console.clear()
                console.print(self.render_summary())
                console.print("\n[dim]Press any key to continue...[/dim]")
                console.input()
            elif key.lower() == 't':
                self.show_timeline()


@click.command()
@click.option('--log-dir', '-d', type=click.Path(exists=True, path_type=Path),
              help='Directory containing LLM logs (default: llm_logs in current directory)')
def main(log_dir: Optional[Path]):
    """Interactive LLM Log Viewer - Browse and analyze LLM interaction logs."""
    if not log_dir:
        # Try to find log directory
        if (Path.cwd() / "llm_logs").exists():
            log_dir = Path.cwd() / "llm_logs"
        elif (Path.cwd().parent / "llm_logs").exists():
            log_dir = Path.cwd().parent / "llm_logs"
        else:
            console.print("[red]Could not find llm_logs directory. Please specify with -d option.[/red]")
            return
    
    viewer = LogViewer(log_dir)
    try:
        viewer.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Viewer closed[/yellow]")


if __name__ == "__main__":
    main() 