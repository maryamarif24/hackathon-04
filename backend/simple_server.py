"""
Simple FastAPI server for testing the chatbot without database dependencies.
This provides mock responses so you can test the chatbot UI.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import re

app = FastAPI(title="Physical AI Textbook API (Mock)")

# Enable CORS - allow all origins for deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change to specific domain in production)
    allow_credentials=False,  # Set to False when using wildcard origins
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

class SourceCitation(BaseModel):
    chunk_id: str
    chapter_id: int
    section_id: str
    section_title: str
    preview_text: str
    relevance_score: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceCitation]
    query_time_ms: float

@app.get("/")
async def root():
    return {
        "name": "Physical AI Textbook API (Mock Mode)",
        "version": "1.0.0",
        "status": "running",
        "mode": "mock",
        "message": "This is a mock server for testing. Set up Qdrant and Neon for full functionality."
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "mode": "mock"}

@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Mock query endpoint that returns sample responses.
    Supports both general questions and context-specific queries based on selected text.
    """
    # Check if this is a context-specific query (contains "Context:" marker)
    question_text = request.question
    context_text = ""

    if "Context:" in question_text:
        parts = question_text.split('\n\nQuestion:')
        if len(parts) >= 2:
            # Extract context from "Context: actual_context_text"
            context_match = re.search(r'Context:\s*(.*)', parts[0])
            if context_match:
                context_text = context_match.group(1).strip()
            question_text = parts[1].strip()

    # If there's context text, provide a response based on the context
    if context_text:
        # Analyze the context and question to provide a relevant response
        answer = generate_context_aware_response(context_text, question_text)
        sources = [
            SourceCitation(
                chunk_id="context-based",
                chapter_id=0,
                section_id="context",
                section_title="Selected Text Context",
                preview_text=context_text[:100] + "..." if len(context_text) > 100 else context_text,
                relevance_score=0.99
            )
        ]
    else:
        # Mock responses based on keywords for general questions
        question_lower = question_text.lower()

        if "physical ai" in question_lower or "what is" in question_lower:
            answer = """Physical AI refers to artificial intelligence systems that interact directly with the physical world through robotic platforms. Unlike traditional AI that operates purely in software, Physical AI combines:

    - **Perception**: Using sensors like cameras, LiDAR, and force sensors to understand the environment
    - **Cognition**: AI models that process sensor data and make decisions in real-time
    - **Action**: Actuators and motors that execute physical tasks

    Physical AI is critical for humanoid robots, autonomous vehicles, and industrial automation systems."""
            sources = [
                SourceCitation(
                    chunk_id="ch1-intro-001",
                    chapter_id=1,
                    section_id="1.1",
                    section_title="Introduction to Physical AI",
                    preview_text="Physical AI represents a paradigm shift in artificial intelligence...",
                    relevance_score=0.95
                )
            ]

        elif "ros" in question_lower or "robot operating system" in question_lower:
            answer = """ROS 2 (Robot Operating System 2) is the industry-standard framework for robot software development. It provides:

    - **Communication Infrastructure**: Nodes, topics, and services for inter-process communication
    - **Hardware Abstraction**: Standardized interfaces for sensors and actuators
    - **Tools and Libraries**: Visualization (RViz), simulation (Gazebo), and debugging tools
    - **Distributed Computing**: Supports multi-robot and cloud-connected systems

    ROS 2 improves upon ROS 1 with real-time capabilities, better security, and multi-platform support."""
            sources = [
                SourceCitation(
                    chunk_id="ch3-ros-001",
                    chapter_id=3,
                    section_id="3.1",
                    section_title="ROS 2 Architecture",
                    preview_text="ROS 2 is built on a distributed middleware called DDS...",
                    relevance_score=0.92
                )
            ]

        elif "humanoid" in question_lower or "robot" in question_lower:
            answer = """Humanoid robotics involves designing robots with human-like form and capabilities. Key components include:

    - **Mechanical Design**: Joints, actuators, and structural elements that mimic human anatomy
    - **Sensors**: Vision systems, tactile sensors, IMUs for balance and perception
    - **Control Systems**: Real-time control loops for walking, manipulation, and interaction
    - **AI Integration**: Vision-language-action models for understanding and responding to commands

    Modern humanoid robots like Tesla Optimus and Boston Dynamics Atlas demonstrate advanced mobility and dexterity."""
            sources = [
                SourceCitation(
                    chunk_id="ch2-humanoid-001",
                    chapter_id=2,
                    section_id="2.1",
                    section_title="Basics of Humanoid Robotics",
                    preview_text="Humanoid robots are designed to replicate human form and function...",
                    relevance_score=0.89
                )
            ]

        elif "vla" in question_lower or "vision-language-action" in question_lower:
            answer = """Vision-Language-Action (VLA) systems are AI models that combine:

    - **Vision**: Processing camera inputs to understand scenes and objects
    - **Language**: Understanding natural language commands and providing explanations
    - **Action**: Generating robot control commands to manipulate objects

    VLA models like RT-2 from Google DeepMind enable robots to understand instructions like "pick up the red cup" and execute the corresponding actions. These systems bridge the gap between human intent and robot execution."""
            sources = [
                SourceCitation(
                    chunk_id="ch5-vla-001",
                    chapter_id=5,
                    section_id="5.1",
                    section_title="Vision-Language-Action Systems",
                    preview_text="VLA systems represent the convergence of computer vision, NLP, and robotics...",
                    relevance_score=0.94
                )
            ]

        else:
            # Generic response
            answer = f"""I can help you understand concepts from the Physical AI and Humanoid Robotics textbook!

    Your question: "{question_text}"

    This textbook covers:
    - Chapter 1: Introduction to Physical AI
    - Chapter 2: Basics of Humanoid Robotics
    - Chapter 3: ROS 2 Fundamentals
    - Chapter 4: Digital Twin Simulation
    - Chapter 5: Vision-Language-Action Systems
    - Chapter 6: Capstone Project

    Try asking about specific topics like "What is Physical AI?", "How does ROS 2 work?", or "Explain VLA systems"."""
            sources = []

    return QueryResponse(
        answer=answer,
        sources=sources,
        query_time_ms=45.2
    )


# Helper function to generate context-aware responses
def generate_context_aware_response(context_text: str, question: str) -> str:
    # Analyze the context and question to provide a relevant response
    context_lower = context_text.lower()
    question_lower = question.lower()

    # Check if the question is asking for explanation of the selected text
    if any(keyword in question_lower for keyword in ['explain', 'what does', 'mean', 'describe']):
        return f"""Based on the selected text: "{context_text}"

This text discusses important concepts in Physical AI and robotics. The selected portion covers key aspects of the topic and provides foundational knowledge. For a more comprehensive understanding, I recommend referring to the relevant sections in the textbook."""

    # Check if the question is asking for more details about something in the context
    if any(keyword in question_lower for keyword in ['more', 'details', 'elaborate', 'further']):
        return f"""The selected text "{context_text}" highlights important concepts in Physical AI and robotics. To elaborate further on this topic:

{get_elaboration_for_context(context_text)}

This builds upon the foundational concepts mentioned in your selected text."""

    # Default context-aware response
    return f"""Based on the selected text: "{context_text}"

Your question "{question}" relates to the concepts mentioned in the selected portion. The text provides context about the topic, and here's what I can tell you:

{get_general_response_for_question(question)}

For more detailed information, please refer to the specific sections in the textbook that contain the selected text."""


# Helper function to get elaboration based on context
def get_elaboration_for_context(context: str) -> str:
    if any(keyword in context.lower() for keyword in ['physical ai', 'embodied ai']):
        return "Physical AI, also known as Embodied AI, represents the integration of artificial intelligence with physical systems. This field focuses on creating AI systems that can interact with and operate in the physical world, combining perception, cognition, and action in real-time."
    if any(keyword in context.lower() for keyword in ['ros', 'robot operating system']):
        return "ROS (Robot Operating System) is a flexible framework for writing robot software. It's a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robotic platforms."
    if any(keyword in context.lower() for keyword in ['humanoid', 'robot']):
        return "Humanoid robots are robots with human-like form and capabilities. They typically feature a head, torso, two arms, and two legs, and may have human-like facial features and the ability to interact with human tools and environments."
    if any(keyword in context.lower() for keyword in ['sensor', 'sensors']):
        return "Sensors in robotics are critical components that enable robots to perceive their environment. Common sensors include cameras for vision, LiDAR for distance measurement, IMUs for orientation, and force/torque sensors for interaction with objects."
    if any(keyword in context.lower() for keyword in ['control', 'controller']):
        return "Robot control systems translate high-level commands into specific motor actions. This involves various control strategies like PID control for precise positioning, motion planning for path generation, and feedback control for error correction."

    return "This topic is fundamental to understanding Physical AI and robotics. The concepts build upon each other to create intelligent systems that can interact with the physical world effectively."


# Helper function to get general response for a question
def get_general_response_for_question(question: str) -> str:
    q = question.lower()

    if 'physical ai' in q:
        return "Physical AI refers to artificial intelligence systems that interact directly with the physical world through robotic platforms. Unlike traditional AI that operates purely in software, Physical AI combines perception, cognition, and action in real-time."
    if any(keyword in q for keyword in ['ros', 'robot operating system']):
        return "ROS (Robot Operating System) is the industry-standard framework for robot software development, providing communication infrastructure, hardware abstraction, and development tools."
    if 'humanoid' in q:
        return "Humanoid robotics involves creating robots with human-like form and capabilities, including mechanical design, sensors, control systems, and AI integration."
    if 'sensor' in q:
        return "Robot sensors enable environmental perception and state estimation, including vision systems, range sensors, and proprioceptive sensors."
    if 'control' in q:
        return "Robot control translates high-level goals into motor commands using various strategies like PID control, MPC, and motion planning algorithms."

    return "This is an important topic in Physical AI and robotics. The textbook covers this in detail with practical examples and applications."

if __name__ == "__main__":
    import uvicorn
    print("Starting Mock Physical AI Textbook API...")
    print("This server provides sample responses for testing the chatbot UI")
    print("API running at: http://localhost:8000")
    print("Docs available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
