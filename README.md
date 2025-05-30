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
   - Generates complete train.py based on context
   - Executes training and captures output
   - Analyzes results and updates tasks in a single LLM call
   - Updates KEY_FINDINGS.md with analysis
3. **Finalization**: Generates comprehensive REPORT.md

## File Structure

```
experiment/
├── IDEA.md           # Your experimental idea (you create this)
├── PLAN.md           # Generated experimental plan
├── tasks.json        # Task tracking in JSON format
├── KEY_FINDINGS.md   # Analysis and insights from each run
├── REPORT.md         # Final report (generated at end)
├── run_1/
│   ├── train.py      # Generated training code
│   └── output.log    # Training output
├── run_2/
│   ├── train.py
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
- `SUMMARIZE_MODEL`: Model for log summarization (default: openai/gpt-4o-mini)
- `CODE_MODEL`: Model for code generation (default: anthropic/claude-3.5-sonnet)
- `REPORT_MODEL`: Model for report generation (default: anthropic/claude-3.5-sonnet)
- `MAX_RUNS`: Maximum number of runs (default: 10)
- `TIMEOUT_SECONDS`: Timeout per run in seconds (default: 7200)
