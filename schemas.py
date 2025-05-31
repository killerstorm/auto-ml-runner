"""JSON schemas for structured LLM outputs."""


INITIAL_TASKS_SCHEMA = {
    "type": "object",
    "properties": {
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Task description"
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": "Task priority"
                    },
                    "dependencies": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of task descriptions this depends on (possibly empty)"
                    }
                },
                "required": ["description", "priority", "dependencies"],
                "additionalProperties": False
            },
            "minItems": 1,
            "description": "List of initial tasks"
        }
    },
    "required": ["tasks"],
    "additionalProperties": False
}

# Separate schemas for split analysis and task updates
ANALYSIS_ONLY_SCHEMA = {
    "type": "object",
    "properties": {
        "analysis": {
            "type": "string",
            "description": "Detailed analysis of the run results"
        },
        "key_findings": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 5,
            "description": "Key findings from this run (max 5)"
        },
        "experiment_state": {
            "type": "object",
            "properties": {
                "experiment_complete": {
                    "type": "boolean",
                    "description": "True if experiment goals have been achieved and final report should be generated"
                },
                "plan_revision_needed": {
                    "type": "boolean",
                    "description": "True if the experimental plan should be revised based on current findings"
                },
                "early_exit_required": {
                    "type": "boolean",
                    "description": "True if no further progress can be made (e.g., missing dependencies, fundamental blockers)"
                },
                "reason": {
                    "type": ["string", "null"],
                    "description": "Explanation for any state flags set to true"
                }
            },
            "required": ["experiment_complete", "plan_revision_needed", "early_exit_required", "reason"],
            "description": "High-level experiment state indicators",
            "additionalProperties": False
        }
    },
    "required": ["analysis", "key_findings", "experiment_state"],
    "additionalProperties": False
}

TASK_UPDATES_SCHEMA = {
    "type": "object",
    "properties": {
        "task_updates": {
            "type": "array",
            "items": {
                "anyOf": [
                    {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "const": "complete",
                                "description": "Mark task as complete"
                            },
                            "task_id": {
                                "type": "string",
                                "description": "Task ID to complete"
                            }
                        },
                        "required": ["action", "task_id"],
                        "additionalProperties": False
                    },
                    {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "const": "add",
                                "description": "Add new task"
                            },
                            "description": {
                                "type": "string",
                                "description": "Task description"
                            },
                            "priority": {
                                "type": "string",
                                "enum": ["high", "medium", "low"],
                                "description": "Task priority"
                            }
                        },
                        "required": ["action", "description", "priority"],
                        "additionalProperties": False
                    },
                    {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "const": "update",
                                "description": "Update task status"
                            },
                            "task_id": {
                                "type": "string",
                                "description": "Task ID to update"
                            },
                            "new_status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "blocked"],
                                "description": "New status for the task"
                            },
                            "notes": {
                                "type": "string",
                                "description": "Optional notes about the status update"
                            }
                        },
                        "required": ["action", "task_id", "new_status", "notes"],
                        "additionalProperties": False
                    },
                    {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "const": "remove",
                                "description": "Remove task"
                            },
                            "task_id": {
                                "type": "string",
                                "description": "Task ID to remove"
                            },
                            "notes": {
                                "type": "string",
                                "description": "Reason for removal"
                            }
                        },
                        "required": ["action", "task_id", "notes"],
                        "additionalProperties": False
                    }
                ]
            },
            "description": "List of task updates to perform"
        }
    },
    "required": ["task_updates"],
    "additionalProperties": False
}