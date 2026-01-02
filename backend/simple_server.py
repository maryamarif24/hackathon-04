"""
Simple FastAPI server for testing the chatbot without database dependencies.
This provides mock responses so you can test the chatbot UI.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

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
    context: str = ""
    use_context_only: bool = False
    chapter_id: Optional[int] = None
    top_k: int = 5

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
    sources: List[SourceCitation]
    chapter_id: Optional[int] = None
    query_time_ms: float
    educational_metadata: Optional[EducationalMetadata] = None

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
    """Mock query endpoint that returns sample responses."""
    question_text = request.question
    context_text = request.context or ""
    use_context_only = request.use_context_only
    chapter_id = request.chapter_id

    # If there's context text, provide a response based on the context
    if context_text:
        answer = generate_context_aware_response(context_text, question_text, use_context_only)
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
            answer = """**Physical AI** represents the convergence of artificial intelligence with physical robotics, enabling machines to perceive, reason about, and interact with the real world.

### Core Principles:
1. **Perception**: Multi-sensor fusion (cameras, LiDAR, tactile, IMUs) for environmental understanding
2. **Cognition**: Real-time AI models for decision-making and planning
3. **Action**: Precise actuation through motors, servos, and end-effectors

### Key Differences from Traditional AI:
- **Traditional AI**: Operates in virtual environments (software, simulations)
- **Physical AI**: Must handle real-world uncertainties, physics, and safety constraints

### Applications:
- Humanoid robots (Tesla Optimus, Boston Dynamics Atlas)
- Autonomous vehicles
- Industrial automation
- Surgical robots
- Warehouse automation (Amazon Robotics)

Physical AI is revolutionizing how machines interact with our physical world!"""
            sources = [
                SourceCitation(
                    chunk_id="ch1-intro-001",
                    chapter_id=1,
                    section_id="1.1",
                    section_title="Introduction to Physical AI",
                    preview_text="Physical AI represents a paradigm shift in artificial intelligence...",
                    relevance_score=0.98
                )
            ]

        elif "ros" in question_lower or "robot operating system" in question_lower:
            answer = """**ROS 2 (Robot Operating System 2)** is the industry-standard middleware for building robot applications.

### Architecture:
- **DDS (Data Distribution Service)**: Real-time pub-sub middleware
- **Quality of Service (QoS)**: Configurable reliability, durability, latency
- **Security**: DDS-Security standard, encrypted communication

### Communication Patterns:
1. **Topics** (Pub/Sub): Best for sensor data, continuous streams
2. **Services** (Request/Response): Best for discrete actions, configuration
3. **Actions** (Goal-based): Best for long-running tasks with feedback

### Essential Tools:
- **RViz2**: 3D visualization of robots and sensor data
- **Gazebo**: Physics simulation
- **Nav2**: Autonomous navigation stack
- **MoveIt 2**: Motion planning for manipulators

### Why ROS 2 > ROS 1:
- Real-time performance
- Multi-robot support
- Better security
- Production-ready (automotive, industrial)"""
            sources = [
                SourceCitation(
                    chunk_id="ch3-ros-001",
                    chapter_id=3,
                    section_id="3.1",
                    section_title="ROS 2 Architecture",
                    preview_text="ROS 2 is built on a distributed middleware called DDS...",
                    relevance_score=0.97
                )
            ]

        elif "humanoid" in question_lower or "robot" in question_lower:
            answer = """**Humanoid Robotics** focuses on creating robots with human-like form and capabilities.

### Mechanical Design:
- **Degrees of Freedom (DOF)**: Modern humanoids have 30+ joints
- **Actuators**: Electric motors, hydraulic systems, or series elastic actuators (SEA)
- **Materials**: Carbon fiber, aluminum alloys for strength-to-weight ratio

### Key Components:
1. **Locomotion System**: Zero Moment Point (ZMP) control for balance
2. **Manipulation**: Dexterous hands with 20+ DOF, force/torque sensors
3. **Sensory Systems**: Vision, proprioception, tactile sensors

### Modern Examples:
- **Tesla Optimus**: 28 DOF, designed for manufacturing
- **Boston Dynamics Atlas**: Parkour, backflips, 360 vision
- **Figure 01**: Commercial applications, OpenAI integration"""
            sources = [
                SourceCitation(
                    chunk_id="ch2-hum-001",
                    chapter_id=2,
                    section_id="2.1",
                    section_title="Humanoid Robotics Fundamentals",
                    preview_text="Humanoid robots mimic human form and function...",
                    relevance_score=0.96
                )
            ]

        elif "vla" in question_lower or "vision-language-action" in question_lower:
            answer = """**Vision-Language-Action (VLA) Systems** unify perception, language understanding, and robot control.

### Architecture:
Vision Input -> Vision Encoder (ViT)
Text Input -> Language Model (T5, PaLM)
    ↓
Fusion Layer (Cross-attention)
    ↓
Policy Network (Transformer)
    ↓
Robot Actions (joint positions/velocities)

### Key Models:
1. **RT-2 (Robotic Transformer 2)** - Google DeepMind:
   - Vision-Language-Action model
   - Zero-shot generalization to new tasks

2. **PaLM-E** - Google:
   - 562B parameter embodied multimodal model
   - Integrates sensor data with language

3. **OpenVLA** - Open-source:
   - 7B parameters, built on LLaMA and DinoV2

### Capabilities:
- Natural language commands
- Multi-step planning
- Generalization to novel objects
- Reasoning about scenes"""
            sources = [
                SourceCitation(
                    chunk_id="ch5-vla-001",
                    chapter_id=5,
                    section_id="5.1",
                    section_title="Vision-Language-Action Systems",
                    preview_text="VLA systems represent the convergence of computer vision, NLP, and robotics...",
                    relevance_score=0.98
                )
            ]

        elif "sensor" in question_lower:
            answer = """**Robot Sensors** enable environmental perception and state estimation.

### Vision Sensors:
- **RGB Cameras**: Color images, object detection
- **Depth Cameras**: Intel RealSense, Azure Kinect (structured light/ToF)
- **Stereo Cameras**: ZED, OAK-D (depth from disparity)

### Range Sensors:
- **2D LiDAR**: SICK, Hokuyo (planar scanning)
- **3D LiDAR**: Velodyne, Ouster (360 point clouds)
- **Ultrasonic**: Short-range obstacle detection

### Inertial/Proprioceptive:
- **IMU**: 6-DOF (accel + gyro) or 9-DOF (+ magnetometer)
- **Joint Encoders**: Absolute or incremental position
- **Force/Torque Sensors**: 6-axis force/torque sensing

### Sensor Fusion:
- Extended Kalman Filter (EKF)
- Particle Filters
- Graph-based SLAM"""
            sources = [
                SourceCitation(
                    chunk_id="ch2-sen-001",
                    chapter_id=2,
                    section_id="2.2",
                    section_title="Robot Sensors",
                    preview_text="Sensors provide robots with perception of their environment...",
                    relevance_score=0.94
                )
            ]

        else:
            answer = """I can help you learn about **Physical AI & Humanoid Robotics**!

### Available Topics:

**Chapter 1: Physical AI**
- What is Physical AI?
- Applications and use cases
- Core components

**Chapter 2: Humanoid Robotics**
- Mechanical design, sensors, actuators
- Modern examples (Tesla Optimus, Atlas)

**Chapter 3: ROS 2**
- Architecture and communication
- Tools (RViz, Gazebo, MoveIt)

**Chapter 4: Simulation**
- Gazebo and Isaac Sim
- Digital twins, sim-to-real transfer

**Chapter 5: VLA Systems**
- RT-2, PaLM-E, OpenVLA
- Natural language control

### Try asking:
- What is Physical AI?
- Tell me about humanoid robots
- How does ROS 2 work?
- Explain VLA systems
- What sensors do robots use?"""
            sources = []

    # Determine educational metadata based on question
    is_simple = len(question_text) < 50
    is_definition = question_text.lower().startswith('what is') or question_text.lower().startswith('what are')
    edu_metadata = EducationalMetadata(
        questionType='definition' if is_definition else 'explanation',
        complexity='simple' if is_simple else 'moderate',
        estimatedWordCount='< 300' if is_simple else '300-500',
        needsStructure=not is_simple
    )

    # Calculate query time
    import time
    start_time = time.time()
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


def generate_context_aware_response(context_text: str, question: str, use_context_only: bool = False) -> str:
    """Generate response based on selected text context."""
    question_lower = question.lower()
    prefix = "Based only on the selected text" if use_context_only else "Based on the selected text"

    # Check if the question is asking for explanation
    if any(keyword in question_lower for keyword in ['explain', 'what does', 'mean', 'describe']):
        return f"""{prefix}: "{context_text}"

This text discusses important concepts in Physical AI and robotics. The selected portion covers key aspects of the topic. Any answer I provide is {'constrained to' if use_context_only else 'enhanced by'} the information provided."""

    # Check if the question is asking for more details
    if any(keyword in question_lower for keyword in ['more', 'details', 'elaborate', 'further']):
        return f"""{prefix}: "{context_text}"

{get_elaboration_for_context(context_text) if not use_context_only else 'My answer is limited to what is contained in this specific text.'}"""

    return f"""{prefix}: "{context_text}"

{'I cannot provide information beyond what is in the selected text.' if use_context_only else 'I can also provide additional context from the textbook.'}"""


def get_elaboration_for_context(context: str) -> str:
    """Get elaboration based on context content."""
    context_lower = context.lower()
    if any(keyword in context_lower for keyword in ['physical ai', 'embodied ai']):
        return "Physical AI, also known as Embodied AI, represents the integration of artificial intelligence with physical systems, combining perception, cognition, and action in real-time."
    if any(keyword in context_lower for keyword in ['ros', 'robot operating system']):
        return "ROS (Robot Operating System) is a flexible framework for writing robot software with tools, libraries, and conventions for creating complex robot behavior."
    if any(keyword in context_lower for keyword in ['humanoid', 'robot']):
        return "Humanoid robots are robots with human-like form and capabilities, featuring a head, torso, two arms, and two legs."
    if any(keyword in context_lower for keyword in ['sensor', 'sensors']):
        return "Sensors in robotics are critical components that enable robots to perceive their environment, including cameras, LiDAR, IMUs, and force/torque sensors."
    if any(keyword in context_lower for keyword in ['control', 'controller']):
        return "Robot control systems translate high-level commands into motor actions using PID control, motion planning, and feedback control."
    return "This topic is fundamental to understanding Physical AI and robotics."


if __name__ == "__main__":
    import uvicorn
    print("^ Starting Mock Physical AI Textbook API...")
    print("This server provides sample responses for testing the chatbot UI")
    print("API running at: http://localhost:8000")
    print("Docs available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
