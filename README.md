# AI SaaS Marketing Research Agent

Initial Python and LangGraph project structure for an AI agent that supports SaaS marketing research.

## Project structure

```text
.
├── agents/      # Agent definitions and role boundaries
├── tools/       # Research tools and external integrations
├── workflow/    # LangGraph state and graph composition
├── prompts/     # System prompts and reusable prompt templates
├── data/        # Research fixtures and local data
├── docs/        # Architecture and project documentation
├── tests/       # Automated tests
├── .env.example # Environment variable template
└── requirements.txt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Agent and workflow business logic will be added in a later iteration.

