---
id: "011"
title: "OpenRouter Gemini 2.5-Flash Update"
stage: "refactor"
date: "2025-12-31"
surface: "agent"
model: "claude-sonnet-4-5-20250929"
feature: "001-rag-chatbot"
branch: "001-rag-chatbot"
user: "maryamarif24"
command: "Update LLM client to use OpenRouter with Gemini 2.5-Flash"
labels: ["backend", "llm", "openrouter", "refactor"]
links:
  spec: "null"
  ticket: "null"
  adr: "null"
  pr: "null"
files:
  - "backend/src/services/openai_service.py"
  - "backend/requirements.txt"
  - "backend/.env.example"
  - "huggingface-spaces/rag-chatbot-backend/simple_server.py"
  - "huggingface-spaces/rag-chatbot-backend/requirements.txt"
tests:
  - "None"
---

## Prompt

Update the LLM client to use OpenRouter with Gemini 2.5-Flash via OpenAI Agent SDK.

Requirements:

1. MODEL CLIENT SETUP:
   - Replace any Gemini API key usage with OpenRouter key (OPENROUTER_API_KEY)
   - Configure base_url to "https://openrouter.ai/api/v1"
   - Use "google-gemini-2.5-flash" as the model name
   - OpenAI client should be initialized with api_key and base_url only

2. CHATBOT BEHAVIOR:
   - Stateless - no conversation history
   - Single-question input, single-answer output
   - Keep ChatResponse schema for output
   - Use fallback response when errors occur

3. FILE CLEANUP:
   - Remove any Gemini-specific files (python-dotenv not needed for config)
   - Keep existing structure but update imports/config

4. OUTPUT FORMAT:
   - Keep ChatResponse schema with answer, sources, query_time_ms, educational_metadata
   - Use Pydantic models for request/response validation
   - Add CORS support for frontend access

5. STRICT RULES:
   - Use OpenAI Agent SDK client configured for OpenRouter
   - Agent, Runner, RunConfig pattern for LLM calls
   - No session/conversation state
   - Proper error handling with fallback responses

6. DELIVERABLES:
   - Updated backend/src/services/openai_service.py
   - Updated huggingface-spaces/rag-chatbot-backend/simple_server.py
   - Updated requirements.txt files
   - No Gemini-specific code remaining

## Response snapshot

Refactored LLM client to use OpenRouter with Gemini 2.5-Flash via OpenAI Agent SDK. Updated both backend and HuggingFace Space servers. Added agents SDK dependency. Fixed typo in simple_server.py (asynccontextcontext -> asynccontext). Committed and pushed changes to git.

## Outcome

- ✅ Impact: Replaced Gemini API with OpenRouter provider, using Gemini 2.5-Flash model via unified API
- 🧪 Tests: None
- 📁 Files: 5 files modified (321 insertions, 365 deletions)
- 🔁 Next prompts: Test the updated chat endpoint, set OPENROUTER_API_KEY on HuggingFace Space
- 🧠 Reflection: OpenRouter provides unified access to multiple LLM providers, simplifying future model switches

## Evaluation notes (flywheel)

- Failure modes observed: Typo in import (asynccontextcontext) - fixed
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Test with real OPENROUTER_API_KEY to verify Gemini 2.5-Flash responses
