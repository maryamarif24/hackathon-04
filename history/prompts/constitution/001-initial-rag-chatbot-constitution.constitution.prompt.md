---
id: 001
title: initial-rag-chatbot-constitution
stage: constitution
date: 2025-12-30
surface: agent
model: claude-sonnet-4.5
feature: none
branch: master
user: unknown
command: /sp.constitution
labels: [constitution, rag-chatbot, governance]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - .specify/memory/constitution.md (created/initialized)
tests:
 - none
---

## Prompt

You are designing an AI-powered Retrieval-Augmented Generation (RAG) chatbot
for an educational textbook titled "Physical AI & Humanoid Robotics".

This constitution applies only to the RAG chatbot system.

Purpose:
- Assist learners in understanding textbook
- Provide accurate, book-grounded answers

Behavior Rules:
- Answer strictly from retrieved book content
- Never hallucinate or add external knowledge
- If information is missing, say "This is not covered in the book"
- Use a clear, educational tone
- Keep answers concise and structured

Retrieval Rules:
- Always retrieve before answering
- Never answer without context
- Cite chapter/section internally (not visible to user)

User Interaction Rules:
- Support full-book questions
- Support questions based ONLY on user-selected text
- Respect chapter context when available

Safety Rules:
- Do not provide instructions outside of book scope
- Do not generate harmful or unsafe robotics instructions

This constitution must be followed for all RAG-related outputs.

## Response snapshot

Successfully created initial constitution v1.0.0 for Physical AI & Humanoid Robotics Textbook RAG chatbot system.

**Constitution includes:**
- Purpose: Assist learners, provide accurate book-grounded answers
- Behavior Rules: Answer only from retrieved content, no hallucination
- Retrieval Rules: Always retrieve before answering, internal citations
- User Interaction Rules: Support full-book and text-selection queries
- Safety Rules: Prevent harmful/unsafe robotics instructions
- Scope: Applies to RAG system (retrieval, generation, display)
- Governance: Amendment procedure, versioning policy, compliance review

**Sync Impact Report:**
- Version: [TEMPLATE] → 1.0.0 (initial ratification)
- Templates checked: All generic SDD templates - no updates needed
- No command files found in .specify/templates/commands/

## Outcome

- ✅ Impact: Constitution established with clear principles for RAG chatbot behavior
- 🧪 Tests: none
- 📁 Files: 1 created (.specify/memory/constitution.md)
- 🔁 Next prompts: Use /sp. commands for further spec-driven development
- 🧠 Reflection: Initial ratification establishes governance for chatbot system while keeping templates generic

## Evaluation notes (flywheel)

- Failure modes observed: none
- Graders run and results (PASS/FAIL): none
- Prompt variant (if applicable): none
- Next experiment (smallest change to try): none
