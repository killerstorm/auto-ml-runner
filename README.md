# Auto ML Runner

NOTE: CURRENTLY LACKING EXIT CRITERIA AND ABORT ON FATAL ERROR.
USE AT YOUR OWN RISK, it WILL RUN AI-GENERATED CODE without SANDBOX OF ANY KIND.

An automated ML experiment runner that uses LLMs to iteratively improve code and achieve experimental goals.

## Features

- **Stage-based execution**: Separate stages for log summarization, code generation, and reporting
- **Configurable models**: Use different models for different stages (e.g., cheaper models for summarization)
- **Full file generation**: Generates complete files instead of patches for simplicity
- **Global state management**: Tracks progress via PLAN.md, TASKS.md, and KEY_FINDINGS.md
- **Automatic retries and error handling**
- **Rich terminal UI with progress indicators**

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure API key:
```bash
cp .env.example .env
# Edit .env and add your OpenRouter API key
```

3. Create your experiment:
```bash
# Create a directory for your experiment
mkdir my-experiment
cd my-experiment

# Write your experimental idea
echo "# My Experiment Idea

Train a neural network to classify MNIST digits with >99% accuracy." > IDEA.md
```

## Usage

Run the experiment:
```bash
python /path/to/runner.py
```

Options:
- `--experiment-dir, -d`: Specify experiment directory (default: current directory)
- `--max-runs, -r`: Override maximum number of runs
- `--resume`: Resume from the last run (not implemented yet)

## How It Works

1. **Initialization**: Reads IDEA.md and generates PLAN.md and initial tasks
2. **Run Loop**: For each run:
   - Creates mechanical summary of previous run's logs (metrics extraction)
   - Generates complete main.py based on context
   - Executes training and captures output
   - Analyzes results and updates tasks in a single LLM call
   - Updates EXPERIMENT_LOG.md with analysis
3. **Finalization**: Generates comprehensive REPORT.md

## File Structure

```
experiment/
├── IDEA.md           # Your experimental idea (you create this)
├── PLAN.md           # Generated experimental plan
├── tasks.json        # Task tracking in JSON format
├── EXPERIMENT_LOG.md   # Analysis and insights from each run
├── REPORT.md         # Final report (generated at end)
├── run_1/
│   ├── main.py      # Generated training code
│   └── output.log    # Training output
├── run_2/
│   ├── main.py
│   └── output.log
└── ...
```

## Viewing Tasks

Use the included viewer to see task status:
```bash
python /path/to/view_tasks.py -d experiment/
```

## Configuration

Environment variables (in .env):
- `OPENROUTER_API_KEY`: Your OpenRouter API key (required)
- `SUMMARIZE_MODEL`: Model for log summarization (default: google/gemini-2.0-flash-lite-001)
- `CODE_MODEL`: Model for code generation (default: google/gemini-2.5-flash-preview-05-20)
- `REPORT_MODEL`: Model for report generation (default: google/gemini-2.5-flash-preview-05-20)
- `ANALYSIS_MODEL`: Model for result analysis (default: openai/o4-mini)
- `MAX_RUNS`: Maximum number of runs (default: 10)
- `TIMEOUT_SECONDS`: Timeout per run in seconds (default: 7200)

Note: `ANALYSIS_MODEL` needs to generate JSON, currently only OpenAI models seem to be able to do this reliably.

### LLM Interaction Logging

Enable detailed logging of all LLM interactions:
- `LLM_ENABLE_LOGGING`: Enable logging (true/false, default: false)
- `LLM_LOG_DIR`: Directory for log files (default: llm_logs)

When enabled, logs include:
- Complete prompts and messages
- Model parameters (temperature, max_tokens, etc.)
- Full responses including token usage
- Timestamps and unique request IDs
- Errors and retry attempts
- **Function context**: Which function made the LLM call
- **Run context**: Associated run number and experiment ID
- **Performance metrics**: Request duration and token usage
- **Call metadata**: Module, line number, and additional context

### Interactive Log Viewer

View and analyze LLM logs with the interactive viewer:

```bash
# View logs from current directory
python /path/to/view_logs.py

# Specify log directory
python /path/to/view_logs.py -d experiment/llm_logs
```

Features:
- **Browse logs**: Navigate through all LLM interactions with arrow keys
- **Filter by**:
  - Function name (e.g., only show `generate_code` calls)
  - Run number (e.g., only show calls from run 3)
  - Experiment ID
- **Sort by**: Timestamp, function, run number, or duration
- **Timeline view**: See the sequence of LLM calls organized by run and function
- **Statistics**: View summary stats including token usage, success rate, and performance
- **Export**: Save filtered logs as JSON/JSONL for further analysis

Keyboard shortcuts:
- `←/→`: Navigate between log entries
- `f`: Filter by function
- `r`: Filter by run number
- `c`: Clear all filters
- `s`: Change sort order
- `d`: Show detailed view of current entry
- `t`: Show timeline view
- `a`: Show statistics
- `h`: Show help
- `q`: Quit

See [README_LOGGING.md](README_LOGGING.md) for detailed logging documentation.
