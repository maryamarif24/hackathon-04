"""
RAG Chatbot Backend API using OpenAI Agents SDK with OpenRouter.

Uses Gemini 2.5-Flash via OpenRouter for answer generation.
"""
import os
import time
import logging
from contextlib import asynccontext
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from openai import OpenAI
from agents import Agent, Runner, RunConfig

logger = logging.getLogger(__name__)

# Initialize OpenRouter client
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
if not OPENROUTER_API_KEY:
    print("WARNING: OPENROUTER_API_KEY not set. Set it via environment variable.")

openai_client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

# Create the RAG chatbot agent
agent = Agent(
    name="RAG Chatbot",
    instructions="""You are an educational AI assistant for the "Physical AI & Humanoid Robotics" textbook.

Your role is to help learners understand textbook content by answering questions accurately and clearly.

CRITICAL RULES:
1. Answer primarily from the provided textbook context when available
2. If context is limited or empty, you may use your general knowledge to help the learner
3. Always be helpful - if you don't know something, explain what you DO know related to the topic
4. Use clear, educational language appropriate for learners
5. Structure answers with bullet points or numbered lists when appropriate
6. Keep answers concise but thorough enough to be educational
7. Maintain a helpful, patient, encouraging tone

When context is provided:
- Use it as the primary source for your answer
- Cite the relevant chapter/section in your answer

When no context is provided:
- Answer based on your knowledge of Physical AI, robotics, ROS 2, simulation, and VLA systems
- Still be helpful and educational

Remember: Your goal is to help learners succeed. Always provide a useful, accurate response.""",
    model="google-gemini-2.5-flash",
)

# Create FastAPI app
@asynccontextcontext
async def lifespan(app: FastAPI):
    print("Starting RAG Chatbot Backend with Gemini 2.5-Flash via OpenRouter")
    print(f"API running at: http://localhost:{PORT}")
    yield

app = FastAPI(
    title="RAG Chatbot Backend",
    description="Physical AI Textbook Chatbot using Gemini 2.5-Flash via OpenRouter",
    version="2.0.0",
    lifespan=lifespan,
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response models
class SourceCitation(BaseModel):
    chunk_id: str
    chapter_id: int
    section_id: str
    section_title: str
    preview_text: str
    relevance_score: float

class EducationalMetadata(BaseModel):
    questionType: str = "general"
    complexity: str = "simple"
    estimatedWordCount: str = "300-500"
    needsStructure: bool = True

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceCitation] = []
    chapter_id: Optional[int] = None
    query_time_ms: float
    educational_metadata: Optional[EducationalMetadata] = None

class QueryRequest(BaseModel):
    question: str
    context: str = ""
    use_context_only: bool = False
    chapter_id: Optional[int] = None

# Endpoints
@app.get("/")
async def root():
    return {
        "name": "RAG Chatbot Backend",
        "version": "2.0.0",
        "status": "running",
        "model": "google-gemini-2.5-flash",
        "provider": "OpenRouter",
        "instructions": "POST /api/query to chat with the bot"
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "model": "google-gemini-2.5-flash"}

@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the RAG chatbot."""
    start_time = time.time()

    question = request.question
    context_text = request.context or ""
    use_context_only = request.use_context_only
    chapter_id = request.chapter_id

    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="Question is required")

    # Build user message
    if context_text:
        user_message = f"""Question: {question}

Context from textbook:
{context_text}

Instructions:
- Answer using the provided context as your primary source
- Cite the chapter/section in your answer
- Be educational and clear
- Structure with lists if appropriate"""
    else:
        user_message = f"""Question: {question}

Instructions:
- Answer based on your knowledge of Physical AI, robotics, ROS 2, simulation, and VLA systems
- Be educational and helpful
- Structure your answer appropriately"""

    try:
        # Run the agent
        result = Runner.run(
            starting_agent=agent,
            input=user_message,
            run_config=RunConfig(
                model_provider=openai_client,
                max_tokens=800,
                temperature=0.3,
            )
        )

        answer = result.final_output.strip()

        # Determine educational metadata
        is_simple = len(question) < 50
        is_definition = question.lower().startswith('what is') or question.lower().startswith('what are')
        edu_metadata = EducationalMetadata(
            questionType='definition' if is_definition else 'explanation',
            complexity='simple' if is_simple else 'moderate',
            estimatedWordCount='< 300' if is_simple else '300-500',
            needsStructure=not is_simple
        )

        # Calculate query time
        query_time_ms = (time.time() - start_time) * 1000

        # Build sources from context if provided
        sources = []
        if context_text:
            sources.append(SourceCitation(
                chunk_id="context-based",
                chapter_id=chapter_id or 0,
                section_id="context",
                section_title="Selected Text Context",
                preview_text=context_text[:100] + "..." if len(context_text) > 100 else context_text,
                relevance_score=0.99
            ))

        return QueryResponse(
            answer=answer,
            sources=sources,
            chapter_id=chapter_id,
            query_time_ms=round(query_time_ms, 1),
            educational_metadata=edu_metadata
        )

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        # Return fallback response
        answer = f"""I'd be happy to help answer your question about "{question}".

As an educational assistant for Physical AI & Humanoid Robotics, I can help you understand:
- Physical AI and embodied intelligence
- Humanoid robot design and components
- ROS 2 architecture and tools
- Digital twin simulation
- Vision-Language-Action (VLA) systems

Please try again!"""

        return QueryResponse(
            answer=answer,
            sources=[],
            chapter_id=chapter_id,
            query_time_ms=(time.time() - start_time) * 1000,
            educational_metadata=EducationalMetadata()
        )

# Get port from environment or use default
PORT = int(os.environ.get("PORT", 7860))

if __name__ == "__main__":
    print(f"Starting RAG Chatbot Backend on port {PORT}")
    print(f"Using Gemini 2.5-Flash via OpenRouter")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
