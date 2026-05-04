# Multi-Agent Project Automation

A local multi-agent software project automation system for demonstrating Agent collaboration, local deployment, and OpenAI-compatible model integration.

## Project Overview

This project shows how multiple AI agents can collaborate to complete software project planning and automation tasks.

The system accepts a software project requirement as input, then coordinates multiple agents to generate:

- Requirement analysis
- System architecture design
- Backend design
- Frontend design
- Test plan
- Project documentation
- QA and risk review

## Key Features

- Multi-agent collaboration workflow
- Local FastAPI web application
- SQLite history storage
- Mock mode without API key
- OpenAI-compatible API support
- Suitable for OpenClaw local deployment practice
- No personal information included in source code

## Agents

The system includes the following agents:

- PlannerAgent: breaks down project requirements
- ArchitectAgent: designs system architecture
- BackendAgent: designs backend APIs and data structures
- FrontendAgent: designs user interface and interaction flow
- TestAgent: creates test plans
- DocAgent: generates project documentation
- QAAgent: checks risks, privacy, and quality issues

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
python app.py
```

Open in browser:

```bash
http://127.0.0.1:8000
```

## Optional Model Configuration

By default, this project runs in mock mode and does not require any API key.

To use a real model, create a `.env` file locally:

```env
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=your_api_key_here
LLM_MODEL=gpt-4o-mini
```

Do not upload your `.env` file to GitHub.

## Privacy Notice

This repository does not include personal information, private keys, API keys, email addresses, phone numbers, or private business data.

All sensitive configuration should be stored locally in `.env`, which is excluded by `.gitignore`.

## License

MIT License
