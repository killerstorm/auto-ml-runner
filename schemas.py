"""JSON schemas for structured LLM outputs."""

LOG_SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["completed", "failed", "timeout", "terminated", "unknown"],
            "description": "Final status of the training run"
        },
        "exit_code": {
            "type": ["integer", "null"],
            "description": "Process exit code if available"
        },
        "metrics": {
            "type": "object",
            "properties": {
                "final_loss": {"type": ["number", "null"]},
                "final_accuracy": {"type": ["number", "null"]},
                "final_val_loss": {"type": ["number", "null"]},
                "final_val_accuracy": {"type": ["number", "null"]},
                "final_test_accuracy": {"type": ["number", "null"]},
                "total_epochs": {"type": ["integer", "null"]},
                "training_time_seconds": {"type": ["number", "null"]}
            },
            "additionalProperties": True,
            "description": "Key metrics from the training run"
        },
        "errors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "message": {"type": "string"},
                    "traceback": {"type": ["string", "null"]}
                },
                "required": ["type", "message"]
            },
            "description": "List of errors encountered"
        },
        "warnings": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of warning messages"
        },
        "gpu_info": {
            "type": ["object", "null"],
            "properties": {
                "used": {"type": "boolean"},
                "memory_used_mb": {"type": ["number", "null"]},
                "cuda_version": {"type": ["string", "null"]}
            },
            "description": "GPU usage information if available"
        },
        "important_output": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 10,
            "description": "Important lines from the output (max 10)"
        }
    },
    "required": ["status", "metrics", "errors", "warnings", "important_output"]
}

ANALYSIS_AND_TASKS_SCHEMA = {
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
            "description": "High-level experiment state indicators"
        },
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
                        "description": "Task ID (required for complete/update)"
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
                ]
            },
            "description": "List of task updates to perform"
        }
    },
    "required": ["analysis", "key_findings", "experiment_state", "task_updates"]
}

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
                "required": ["description", "priority"]
            },
            "minItems": 1,
            "description": "List of initial tasks"
        }
    },
    "required": ["tasks"]
}