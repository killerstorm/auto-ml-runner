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
                        "description": "List of task descriptions this depends on"
                    }
                },
                "required": ["description", "priority"],
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
                    "type": "string",
                    "description": "Explanation for any state flags set to true"
                }
            },
            "required": ["experiment_complete", "plan_revision_needed", "early_exit_required"],
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
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["complete", "add", "update", "remove"],
                        "description": "Action to perform on tasks"
                    },
                    "task_id": {
                        "type": "string",
                        "description": "Task ID (required for complete/update/remove)"
                    },
                    "description": {
                        "type": "string",
                        "description": "Task description (required for add)"
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": "Priority (required for add)"
                    },
                    "notes": {
                        "type": "string",
                        "description": "Notes about the task update"
                    },
                    "new_status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "blocked"],
                        "description": "New status (required for update action)"
                    }
                },
                "required": ["action"],
                "allOf": [
                    {
                        "if": {"properties": {"action": {"const": "complete"}}},
                        "then": {"required": ["task_id"]}
                    },
                    {
                        "if": {"properties": {"action": {"const": "remove"}}},
                        "then": {"required": ["task_id"]}
                    },
                    {
                        "if": {"properties": {"action": {"const": "add"}}},
                        "then": {"required": ["description", "priority"]}
                    },
                    {
                        "if": {"properties": {"action": {"const": "update"}}},
                        "then": {"required": ["task_id", "new_status"]}
                    }
                ],
                "additionalProperties": False
            },
            "description": "List of task updates to perform"
        }
    },
    "required": ["task_updates"],
    "additionalProperties": False
}