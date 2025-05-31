"""Task management with JSON-based storage."""
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class Task:
    """Represents a single task."""
    id: str
    description: str
    status: str  # pending, in_progress, completed, blocked
    priority: str  # high, medium, low
    created_at: str
    updated_at: str
    completed_at: Optional[str] = None
    notes: Optional[str] = None
    dependencies: Optional[List[str]] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Task':
        """Create from dictionary."""
        if 'status' not in data:
            data['status'] = 'pending'
        return cls(**data)


class TaskManager:
    """Manages tasks with JSON persistence."""
    
    def __init__(self, tasks_file: Path):
        self.tasks_file = tasks_file
        self.tasks: List[Task] = []
        self.load_tasks()
    
    def load_tasks(self):
        """Load tasks from JSON file."""
        if self.tasks_file.exists():
            try:
                with open(self.tasks_file, 'r') as f:
                    data = json.load(f)
                    self.tasks = [Task.from_dict(t) for t in data.get('tasks', [])]
            except Exception as e:
                # If file is corrupted, start fresh
                print(f"Error loading tasks from {self.tasks_file}: {e}")
                self.tasks = []
        else:
            self.tasks = []
    
    def save_tasks(self):
        """Save tasks to JSON file."""
        data = {
            'tasks': [t.to_dict() for t in self.tasks],
            'last_updated': datetime.now().isoformat()
        }
        with open(self.tasks_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add_task(self, description: str, priority: str = 'medium', 
                 dependencies: Optional[List[str]] = None) -> Task:
        """Add a new task."""
        task_id = f"task_{len(self.tasks) + 1:03d}"
        now = datetime.now().isoformat()
        
        task = Task(
            id=task_id,
            description=description,
            status='pending',
            priority=priority,
            created_at=now,
            updated_at=now,
            dependencies=dependencies
        )
        
        self.tasks.append(task)
        self.save_tasks()
        return task
    
    def update_task(self, task_id: str, **kwargs) -> Optional[Task]:
        """Update a task's attributes."""
        for task in self.tasks:
            if task.id == task_id:
                for key, value in kwargs.items():
                    if hasattr(task, key):
                        setattr(task, key, value)
                task.updated_at = datetime.now().isoformat()
                
                # Set completed_at if status changed to completed
                if kwargs.get('status') == 'completed' and not task.completed_at:
                    task.completed_at = task.updated_at
                
                self.save_tasks()
                return task
        return None
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None
    
    def get_tasks_by_status(self, status: str) -> List[Task]:
        """Get all tasks with a specific status."""
        return [t for t in self.tasks if t.status == status]
    
    def get_pending_tasks(self) -> List[Task]:
        """Get all pending tasks sorted by priority."""
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        pending = self.get_tasks_by_status('pending')
        return sorted(pending, key=lambda t: priority_order.get(t.priority, 3))
    
    def mark_completed(self, task_id: str, notes: Optional[str] = None) -> bool:
        """Mark a task as completed."""
        task = self.update_task(task_id, status='completed', notes=notes)
        return task is not None
    
    def remove_task(self, task_id: str) -> bool:
        """Remove a task from the list."""
        for i, task in enumerate(self.tasks):
            if task.id == task_id:
                self.tasks.pop(i)
                self.save_tasks()
                return True
        return False
    
    def to_markdown(self) -> str:
        """Export tasks as markdown for display."""
        md = "# Tasks\n\n"
        
        # Group by status
        status_groups = {
            'in_progress': 'In Progress',
            'pending': 'Pending',
            'completed': 'Completed',
            'blocked': 'Blocked'
        }
        
        for status, title in status_groups.items():
            tasks = self.get_tasks_by_status(status)
            if tasks:
                md += f"## {title}\n\n"
                for task in tasks:
                    checkbox = "x" if status == 'completed' else " "
                    priority_emoji = {
                        'high': '🔴',
                        'medium': '🟡', 
                        'low': '🟢'
                    }.get(task.priority, '')
                    
                    md += f"- [{checkbox}] {priority_emoji} {task.description}"
                    if task.notes:
                        md += f" *({task.notes})*"
                    md += "\n"
                md += "\n"
        
        return md
    
    def get_structured_state(self) -> Dict:
        """Get the complete task state as structured data."""
        return {
            'total_tasks': len(self.tasks),
            'pending': len(self.get_tasks_by_status('pending')),
            'in_progress': len(self.get_tasks_by_status('in_progress')),
            'completed': len(self.get_tasks_by_status('completed')),
            'blocked': len(self.get_tasks_by_status('blocked')),
            'tasks': [t.to_dict() for t in self.tasks]
        }