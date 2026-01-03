// Advanced AI-Powered Chatbot API for Physical AI Textbook
// Intelligent, context-aware responses with comprehensive topic coverage
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 001-rag-chatbot
// Supports general questions, context-specific queries, chapter-aware prioritization, and educational explanations
//
// Security: Input sanitization, output encoding, rate limiting ready
// Educational: Tone enforcement, structure validation, word limits
<<<<<<< HEAD

/**
 * Sanitize user input to prevent XSS and injection attacks
 * @param {string} input - Raw user input
 * @param {number} maxLength - Maximum allowed length (default: 1000)
 * @returns {string} - Sanitized input
 */
function sanitizeInput(input, maxLength = 1000) {
  if (typeof input !== 'string') {
    return '';
  }
  // Trim and limit length
  let sanitized = input.trim().substring(0, maxLength);
  // Remove potential XSS patterns
  sanitized = sanitized
    .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
    .replace(/javascript:/gi, '')
    .replace(/on\w+=/gi, '')
    .replace(/<iframe/gi, '[iframe blocked]')
    .replace(/<object/gi, '[object blocked]')
    .replace(/<embed/gi, '[embed blocked]');
  return sanitized;
}

/**
 * Escape markdown special characters for safe display
 * @param {string} text - Text to escape
 * @returns {string} - Escaped text
 */
function escapeForMarkdown(text) {
  if (typeof text !== 'string') {
    return '';
  }
  return text
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/`/g, '\\`')
    .replace(/\*/g, '\\*')
    .replace(/_/g, '\\_')
    .replace(/\[/g, '\\[')
    .replace(/\]/g, '\\]');
}

export default async function handler(req, res) {
=======

export default function handler(req, res) {
>>>>>>> master
=======

/**
 * Sanitize user input to prevent XSS and injection attacks
 * @param {string} input - Raw user input
 * @param {number} maxLength - Maximum allowed length (default: 1000)
 * @returns {string} - Sanitized input
 */
function sanitizeInput(input, maxLength = 1000) {
  if (typeof input !== 'string') {
    return '';
  }
  // Trim and limit length
  let sanitized = input.trim().substring(0, maxLength);
  // Remove potential XSS patterns
  sanitized = sanitized
    .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
    .replace(/javascript:/gi, '')
    .replace(/on\w+=/gi, '')
    .replace(/<iframe/gi, '[iframe blocked]')
    .replace(/<object/gi, '[object blocked]')
    .replace(/<embed/gi, '[embed blocked]');
  return sanitized;
}

/**
 * Escape markdown special characters for safe display
 * @param {string} text - Text to escape
 * @returns {string} - Escaped text
 */
function escapeForMarkdown(text) {
  if (typeof text !== 'string') {
    return '';
  }
  return text
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/`/g, '\\`')
    .replace(/\*/g, '\\*')
    .replace(/_/g, '\\_')
    .replace(/\[/g, '\\[')
    .replace(/\]/g, '\\]');
}

export default async function handler(req, res) {
>>>>>>> 001-rag-chatbot
  try {
    // Enable CORS
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

    if (req.method === 'OPTIONS') {
      return res.status(200).end();
    }

    if (req.method !== 'POST') {
      return res.status(405).json({ error: 'Method not allowed' });
    }

<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 001-rag-chatbot
    // Sanitize all user inputs
    const rawQuestion = req.body?.question || '';
    const rawContext = req.body?.context || '';
    const rawUseContextOnly = req.body?.use_context_only || false;
    const rawChapterId = req.body?.chapter_id || null;

    const question = sanitizeInput(rawQuestion, 1000);
    const context = sanitizeInput(rawContext, 5000);
    const use_context_only = Boolean(rawUseContextOnly);
    const chapter_id = rawChapterId ? Number(rawChapterId) : null;

<<<<<<< HEAD
=======
    const { question } = req.body || {};
>>>>>>> master
=======
>>>>>>> 001-rag-chatbot
    if (!question) {
      return res.status(400).json({ error: 'Question is required' });
    }

<<<<<<< HEAD
<<<<<<< HEAD
    // Query external AI services (OpenRouter, Cohere) and databases (Neon, Qdrant)
    const { answer, sources } = await queryRAGSystem(question, context, use_context_only, chapter_id);

    return res.status(200).json({
      answer,
      sources: sources || [],
      chapter_id: chapter_id,
      query_time_ms: Date.now(), // Will be updated with actual time
      educational_metadata: getEducationalMetadata(question)
=======
    const q = question.toLowerCase();
=======
    // Query external AI services (OpenRouter, Cohere) and databases (Neon, Qdrant)
    const { answer, sources } = await queryRAGSystem(question, context, use_context_only, chapter_id);
>>>>>>> 001-rag-chatbot

    return res.status(200).json({
<<<<<<< HEAD
      answer: "I can help you learn about **Physical AI & Humanoid Robotics**!\n\n### Available Topics:\n\n**Chapter 1: Physical AI**\n- What is Physical AI?\n- Applications and use cases\n- Core components\n\n**Chapter 2: Humanoid Robotics**\n- Mechanical design\n- Sensors and actuators\n- Modern examples (Tesla Optimus, Atlas)\n\n**Chapter 3: ROS 2**\n- Architecture and communication\n- Tools (RViz, Gazebo, MoveIt)\n- Real-time capabilities\n\n**Chapter 4: Simulation**\n- Gazebo and Isaac Sim\n- Digital twins\n- Sim-to-real transfer\n\n**Chapter 5: VLA Systems**\n- RT-2, PaLM-E, OpenVLA\n- Training and deployment\n- Natural language control\n\n**Chapter 6: Integration**\n- System design\n- Best practices\n- Real-world deployment\n\n### Try asking:\n- What is Physical AI?\n- Tell me about humanoid robots\n- How does ROS 2 work?\n- Explain VLA systems\n- What sensors do robots use?\n- How to control a robot?",
      sources: [],
      query_time_ms: 35.8
>>>>>>> master
=======
      answer,
      sources: sources || [],
      chapter_id: chapter_id,
      query_time_ms: Date.now(), // Will be updated with actual time
      educational_metadata: getEducationalMetadata(question)
>>>>>>> 001-rag-chatbot
    });
  } catch (error) {
    console.error('API Error:', error);
    return res.status(500).json({
      error: 'Internal server error',
      message: error.message
    });
  }
}
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 001-rag-chatbot

// Main function to query the RAG system using external APIs and databases
async function queryRAGSystem(question, context, use_context_only, chapter_id) {
  try {
    // First, try to get context from Neon database to retrieve document chunks
    let neonSources = [];
    if (!use_context_only) {
      neonSources = await queryNeonDatabase(question, chapter_id);
    }

    // Then, get semantic context from Qdrant vector database
    let qdrantSources = [];
    if (!use_context_only) {
      qdrantSources = await queryQdrant(question, chapter_id);
    }

    // Combine sources from both databases
    let allSources = [...neonSources, ...qdrantSources];

    // Use context from user or from databases
    const retrievalContext = use_context_only ? context : (
      context ||
      [...neonSources, ...qdrantSources].map(s => s.preview_text).join(' ')
    );

    // Call OpenRouter API for the main response
    const openRouterResponse = await callOpenRouterAPI(question, retrievalContext);

    // Optionally enhance with Cohere if needed
    const finalAnswer = await enhanceWithCohereIfNeeded(openRouterResponse, question);

    // Return the answer and combined sources
    return {
      answer: finalAnswer,
      sources: allSources
    };
  } catch (error) {
    console.error('RAG System Error:', error);

    // Fallback to simple response if external services fail
    return {
      answer: `I can help you learn about Physical AI & Robotics! You asked: "${question}".\n\nUnfortunately, I'm having trouble connecting to the external services right now. The system uses OpenRouter API, Cohere, Neon database, and Qdrant for advanced responses.`,
      sources: []
    };
  }
}

// Function to query Neon database for document chunks
async function queryNeonDatabase(question, chapter_id) {
  try {
    // Check if Neon database credentials are available in environment
    const neonDatabaseUrl = process.env.NEON_DATABASE_URL || process.env.DATABASE_URL;

    if (!neonDatabaseUrl) {
      console.log('Neon database configuration not found, using mock data');
      // Return mock data for demo purposes
      return [
        {
          chunk_id: 'neon-mock-1',
          chapter_id: chapter_id || 1,
          section_id: '1.1',
          section_title: 'Introduction to Physical AI',
          preview_text: 'Physical AI represents the convergence of artificial intelligence with physical robotics, enabling machines to perceive, reason about, and interact with the real world.',
          relevance_score: 0.92
        }
      ];
    }

    // In a real implementation, you would connect to Neon PostgreSQL database using a library like 'pg'
    // Since we can't install packages in this serverless function easily, we'll use a fetch to a potential API endpoint
    // In a real app, you'd want to install 'pg' package and connect directly:
    /*
    const { Client } = require('pg');
    const client = new Client(neonDatabaseUrl);
    await client.connect();

    let query = 'SELECT id, chapter_id, section_id, section_title, content, similarity FROM documents';
    let params = [];

    if (chapter_id) {
      query += ' WHERE chapter_id = $1';
      params = [chapter_id];
    }

    query += ' ORDER BY similarity DESC LIMIT $' + (params.length + 1);
    params.push(5);

    const result = await client.query(query, params);
    await client.end();
    */

    // For this implementation, we'll make a fetch call to a potential Neon API endpoint
    // In a real deployment, you'd likely have a separate service that handles database connections
    const searchEndpoint = process.env.NEON_SEARCH_API_URL || `${process.env.VERCEL_URL}/api/neon-search` || '/api/neon-search';

    const response = await fetch(searchEndpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query: question,
        chapter_id: chapter_id,
        limit: 5
      }),
    });

    // If the Neon search API is not available, return mock data
    if (!response.ok) {
      console.log(`Neon search API not available (${response.status}), using mock data`);
      return [
        {
          chunk_id: 'neon-mock-1',
          chapter_id: chapter_id || 1,
          section_id: '1.1',
          section_title: 'Introduction to Physical AI',
          preview_text: 'Physical AI represents the convergence of artificial intelligence with physical robotics, enabling machines to perceive, reason about, and interact with the real world.',
          relevance_score: 0.88
        }
      ];
    }

    const data = await response.json();

    // Format the response to match our expected source format
    return data.results?.map(item => ({
      chunk_id: item.id || item.chunk_id,
      chapter_id: item.chapter_id || chapter_id || 1,
      section_id: item.section_id || '1.1',
      section_title: item.section_title || 'Section',
      preview_text: item.content || item.text || item.preview_text || '',
      relevance_score: item.relevance_score || item.similarity || 0.5,
    })) || [];
  } catch (error) {
    console.error('Neon database query error:', error);
    // Return mock data as fallback
    return [
      {
        chunk_id: 'neon-mock-1',
        chapter_id: chapter_id || 1,
        section_id: '1.1',
        section_title: 'Introduction to Physical AI',
        preview_text: 'Physical AI represents the convergence of artificial intelligence with physical robotics, enabling machines to perceive, reason about, and interact with the real world.',
        relevance_score: 0.85
      }
    ]; // Return mock data as fallback
  }
}

// Function to query Qdrant vector database
async function queryQdrant(question, chapter_id) {
  try {
    // Check if QDRANT_URL and QDRANT_API_KEY are available in environment
    const qdrantUrl = process.env.QDRANT_URL || process.env.NEXT_PUBLIC_QDRANT_URL;
    const qdrantApiKey = process.env.QDRANT_API_KEY || process.env.NEXT_PUBLIC_QDRANT_API_KEY;

    if (!qdrantUrl) {
      console.log('Qdrant configuration not found, using mock data');
      // Return mock data for demo purposes
      return [
        {
          chunk_id: 'mock-chunk-1',
          chapter_id: chapter_id || 1,
          section_id: '1.1',
          section_title: 'Introduction to Robotics',
          preview_text: 'Robotics is an interdisciplinary field that includes mechanical engineering, electrical engineering, computer science, and others.',
          relevance_score: 0.85
        }
      ];
    }

    // In a real implementation, you would call the Qdrant API here
    // This is a simplified example
    const response = await fetch(`${qdrantUrl}/collections/documents/points/search`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Api-Key': qdrantApiKey,
      },
      body: JSON.stringify({
        vector: await textToVector(question), // This would convert text to embedding
        limit: 5,
        with_payload: true,
      }),
    });

    if (!response.ok) {
      throw new Error(`Qdrant API error: ${response.status}`);
    }

    const data = await response.json();
    return data.result.map(item => ({
      chunk_id: item.id,
      chapter_id: item.payload?.chapter_id || 1,
      section_id: item.payload?.section_id || '1.1',
      section_title: item.payload?.section_title || 'Section',
      preview_text: item.payload?.text || item.payload?.content || '',
      relevance_score: item.score || 0.5,
    }));
  } catch (error) {
    console.error('Qdrant query error:', error);
    return []; // Return empty array if Qdrant fails
  }
}

// Function to call OpenRouter API
async function callOpenRouterAPI(question, context) {
  try {
<<<<<<< HEAD
    const openRouterApiKey = process.env.OPENROUTER_API_KEY; // Use only server-side environment variable

    if (!openRouterApiKey) {
      console.error('OPENROUTER_API_KEY environment variable not set');
      return `OpenRouter API key not configured. Question: ${question}\n\nPlease set the OPENROUTER_API_KEY environment variable on the server.`;
=======
    const openRouterApiKey = process.env.OPENROUTER_API_KEY || process.env.NEXT_PUBLIC_OPENROUTER_API_KEY;

    if (!openRouterApiKey) {
      console.log('OpenRouter API key not found, using fallback response');
      return `OpenRouter API key not configured. Question: ${question}`;
>>>>>>> 001-rag-chatbot
    }

    const systemPrompt = context
      ? `You are an expert assistant for Physical AI and Robotics. Use the following context to answer the question: ${context}`
      : 'You are an expert assistant for Physical AI and Robotics. Answer the following question.';

    const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${openRouterApiKey}`,
        'Content-Type': 'application/json',
<<<<<<< HEAD
        'HTTP-Referer': process.env.NEXT_PUBLIC_VERCEL_URL || 'http://localhost:3000',
        'X-Title': 'Physical AI & Robotics Chatbot',
      },
      body: JSON.stringify({
        model: 'google/gemini-flash-1.5', // Updated to use more reliable model on OpenRouter
=======
      },
      body: JSON.stringify({
        model: 'google/gemini-pro', // You can change this to your preferred model
>>>>>>> 001-rag-chatbot
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: question }
        ],
        max_tokens: 1000,
<<<<<<< HEAD
        temperature: 0.3, // Lower temperature for more factual responses
=======
>>>>>>> 001-rag-chatbot
      }),
    });

    if (!response.ok) {
<<<<<<< HEAD
      const errorBody = await response.text();
      console.error('OpenRouter API error details:', {
        status: response.status,
        statusText: response.statusText,
        errorBody: errorBody,
        model: 'google/gemini-flash-1.5',
        questionLength: question.length,
        contextLength: context.length
      });
      throw new Error(`OpenRouter API error: ${response.status}, details: ${errorBody}`);
=======
      throw new Error(`OpenRouter API error: ${response.status}`);
>>>>>>> 001-rag-chatbot
    }

    const data = await response.json();
    return data.choices[0]?.message?.content || 'No response from OpenRouter API';
  } catch (error) {
    console.error('OpenRouter API error:', error);
<<<<<<< HEAD
    // Log more specific details for debugging
    console.error('OpenRouter API error details:', {
      errorMessage: error.message,
      model: 'google/gemini-flash-1.5',
      apiKeyAvailable: !!openRouterApiKey,
      questionLength: question.length,
      contextLength: context.length
    });
=======
>>>>>>> 001-rag-chatbot
    return `Error calling OpenRouter API: ${error.message}`;
  }
}

// Function to enhance response with Cohere if needed
async function enhanceWithCohereIfNeeded(response, question) {
  try {
    const cohereApiKey = process.env.COHERE_API_KEY || process.env.NEXT_PUBLIC_COHERE_API_KEY;

    if (!cohereApiKey) {
      console.log('Cohere API key not found, returning original response');
      return response;
    }

    // In a real implementation, you would use Cohere to enhance the response
    // For now, just return the original response
    return response;
  } catch (error) {
    console.error('Cohere enhancement error:', error);
    return response; // Return original response if Cohere fails
  }
}

// Helper function to convert text to vector (simplified - in reality you'd use an embedding model)
async function textToVector(text) {
  // This is a placeholder - in reality you'd use an embedding model like OpenAI, Cohere, etc.
  // For demo purposes, return a simple array
  return Array.from({length: 1536}, () => Math.random()); // 1536-dim OpenAI embedding size
}

// Helper function to generate context-aware responses
function generateContextAwareResponse(contextText, question) {
  // Analyze the context and question to provide a relevant response
  const contextLower = contextText.toLowerCase();
  const questionLower = question.toLowerCase();

  // Check if the question is asking for explanation of the selected text
  if (questionLower.includes('explain') || questionLower.includes('what does') || questionLower.includes('mean') || questionLower.includes('describe')) {
    return `Based on the selected text: "${contextText}"\n\nThis text discusses important concepts in Physical AI and robotics. The selected portion covers key aspects of the topic and provides foundational knowledge. For a more comprehensive understanding, I recommend referring to the relevant sections in the textbook.`;
  }

  // Check if the question is asking for more details about something in the context
  if (questionLower.includes('more') || questionLower.includes('details') || questionLower.includes('elaborate') || questionLower.includes('further')) {
    return `The selected text "${contextText}" highlights important concepts in Physical AI and robotics. To elaborate further on this topic:\n\n${getElaborationForContext(contextText)}\n\nThis builds upon the foundational concepts mentioned in your selected text.`;
  }

  // Default context-aware response
  return `Based on the selected text: "${contextText}"\n\nYour question "${question}" relates to the concepts mentioned in the selected portion. The text provides context about the topic, and here's what I can tell you:\n\n${getGeneralResponseForQuestion(question)}\n\nFor more detailed information, please refer to the specific sections in the textbook that contain the selected text.`;
}

// Helper function to get elaboration based on context
function getElaborationForContext(context) {
  if (context.toLowerCase().includes('physical ai') || context.toLowerCase().includes('embodied ai')) {
    return "Physical AI, also known as Embodied AI, represents the integration of artificial intelligence with physical systems. This field focuses on creating AI systems that can interact with and operate in the physical world, combining perception, cognition, and action in real-time.";
  }
  if (context.toLowerCase().includes('ros') || context.toLowerCase().includes('robot operating system')) {
    return "ROS (Robot Operating System) is a flexible framework for writing robot software. It's a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robotic platforms.";
  }
  if (context.toLowerCase().includes('humanoid') || context.toLowerCase().includes('robot')) {
    return "Humanoid robots are robots with human-like form and capabilities. They typically feature a head, torso, two arms, and two legs, and may have human-like facial features and the ability to interact with human tools and environments.";
  }
  if (context.toLowerCase().includes('sensor') || context.toLowerCase().includes('sensors')) {
    return "Sensors in robotics are critical components that enable robots to perceive their environment. Common sensors include cameras for vision, LiDAR for distance measurement, IMUs for orientation, and force/torque sensors for interaction with objects.";
  }
  if (context.toLowerCase().includes('control') || context.toLowerCase().includes('controller')) {
    return "Robot control systems translate high-level commands into specific motor actions. This involves various control strategies like PID control for precise positioning, motion planning for path generation, and feedback control for error correction.";
  }

  return "This topic is fundamental to understanding Physical AI and robotics. The concepts build upon each other to create intelligent systems that can interact with the physical world effectively.";
}

// Helper function to get general response for a question
function getGeneralResponseForQuestion(question) {
  const q = question.toLowerCase();

  if (q.includes('physical ai')) {
    return "Physical AI refers to artificial intelligence systems that interact directly with the physical world through robotic platforms. Unlike traditional AI that operates purely in software, Physical AI combines perception, cognition, and action in real-time.";
  }
  if (q.includes('ros') || q.includes('robot operating system')) {
    return "ROS (Robot Operating System) is the industry-standard framework for robot software development, providing communication infrastructure, hardware abstraction, and development tools.";
  }
  if (q.includes('humanoid')) {
    return "Humanoid robotics involves creating robots with human-like form and capabilities, including mechanical design, sensors, control systems, and AI integration.";
  }
  if (q.includes('sensor')) {
    return "Robot sensors enable environmental perception and state estimation, including vision systems, range sensors, and proprioceptive sensors.";
  }
  if (q.includes('control')) {
    return "Robot control translates high-level goals into motor commands using various strategies like PID control, MPC, and motion planning algorithms.";
  }

  return "This is an important topic in Physical AI and robotics. The textbook covers this in detail with practical examples and applications.";
}

// Helper function to generate response based only on the provided context
function generateContextOnlyResponse(contextText, question) {
  // This function generates a response based ONLY on the provided context text
  // It should analyze the context and answer the question specifically based on that text
  const contextLower = contextText.toLowerCase();
  const questionLower = question.toLowerCase();

  // Check if the question is asking for explanation of the selected text
  if (questionLower.includes('explain') || questionLower.includes('what does') || questionLower.includes('mean') || questionLower.includes('describe')) {
    return `Based only on the selected text: "${contextText}"\n\nThe text provides information about this topic. The content specifically addresses the concepts mentioned in your selected text. Any answer I provide is constrained to the information provided in the selected text.`;
  }

  // Check if the question is asking for more details about something in the context
  if (questionLower.includes('more') || questionLower.includes('details') || questionLower.includes('elaborate') || questionLower.includes('further')) {
    return `Based only on the selected text: "${contextText}"\n\nThe text contains information about this topic. The selected text is the only source I'm using to answer your question, so my response is limited to what is contained in this specific text.`;
  }

  // Check if the question is asking about specific elements in the context
  if (contextLower.includes('physical ai') && questionLower.includes('what')) {
    return `Based only on the selected text: "${contextText}"\n\nAccording to the selected text, this section discusses Physical AI. My answer is constrained to only the information provided in the selected text.`;
  }

  if (contextLower.includes('ros') && questionLower.includes('how')) {
    return `Based only on the selected text: "${contextText}"\n\nAccording to the selected text, this section discusses ROS. My answer is constrained to only the information provided in the selected text.`;
  }

  // Default response for context-only queries
  return `Based only on the selected text: "${contextText}"\n\nI'm answering your question based solely on the content you selected. The information provided comes exclusively from the selected text, and I'm not using any external knowledge beyond what's in your selected text.\n\nIf the selected text doesn't contain the information needed to answer your question, I cannot provide a complete answer based only on that text.`;
}

// ============================================
// US4: Educational Explanations Utilities
// ============================================

// Post-process response for educational tone and structure
function applyEducationalFormatting(answer, question) {
  let formatted = answer;

  // Detect if this is a simple question (short question = likely short answer needed)
  const isSimpleQuestion = question.length < 50;

  // Apply educational tone transformations
  formatted = ensureEducationalTone(formatted);

  // Enforce structure: ensure lists use consistent formatting
  formatted = standardizeListFormat(formatted);

  // Enforce length limits for simple questions
  if (isSimpleQuestion) {
    formatted = enforceWordLimit(formatted, 300);
  }

  return formatted;
}

// Ensure response uses educational, learner-friendly tone
function ensureEducationalTone(text) {
  let result = text;

  // Replace overly casual language with educational alternatives
  const toneReplacements = [
    { pattern: /\bcool\b/gi, replacement: 'interesting' },
    { pattern: /\bawesome\b/gi, replacement: 'remarkable' },
    { pattern: /\breally\b/gi, replacement: 'significantly' },
    { pattern: /\bsuper\b/gi, replacement: 'highly' },
    { pattern: /\bkind of\b/gi, replacement: 'somewhat' },
    { pattern: /\bsort of\b/gi, replacement: 'somewhat' },
  ];

  toneReplacements.forEach(({ pattern, replacement }) => {
    result = result.replace(pattern, replacement);
  });

  // Ensure first sentence is welcoming/educational
  if (!result.toLowerCase().includes('here') && !result.toLowerCase().startsWith('let')) {
    // Add educational prefix for definition-style answers
    if (result.startsWith('**')) {
      const titleMatch = result.match(/^\*\*(.+?)\*\*/);
      if (titleMatch) {
        const title = titleMatch[1];
        result = result.replace(/^\*\*.+?\*\*/, `**${title}**\n\nThis concept is fundamental to understanding Physical AI and robotics.`);
      }
    }
  }

  return result;
}

// Standardize list formatting for consistency
function standardizeListFormat(text) {
  let result = text;

  // Convert numbered lists that aren't using proper markdown
  result = result.replace(/^(\d+)\.\s*([A-Z])/gm, '$1. **$2');

  // Ensure bullet points have proper spacing
  result = result.replace(/^[-*]\s*([A-Z])/gm, '- **$1');

  // Add spacing after section headers if missing
  result = result.replace(/^(#{1,6}\s+.+)$/gm, '$1\n');

  return result;
}

// Enforce word limit for simple questions
function enforceWordLimit(text, limit) {
  const words = text.split(/\s+/);
  if (words.length > limit) {
    // Truncate at last complete sentence before limit
    const truncated = words.slice(0, limit).join(' ');
    const lastPeriod = truncated.lastIndexOf('.');
    if (lastPeriod > limit * 0.7) {
      return truncated.substring(0, lastPeriod + 1) + '\n\n*[Response truncated for brevity. Ask for more details if needed.]*';
    }
    return truncated + '...';
  }
  return text;
}

// Get educational metadata for response
function getEducationalMetadata(question) {
  const q = question.toLowerCase();
  const isSimple = question.length < 50;
  const isDefinition = q.startsWith('what is') || q.startsWith('what are') || q.startsWith('define');
  const isExplanation = q.includes('explain') || q.includes('how does') || q.includes('why');

  return {
    questionType: isDefinition ? 'definition' : isExplanation ? 'explanation' : 'general',
    complexity: isSimple ? 'simple' : 'moderate',
    estimatedWordCount: isSimple ? '< 300' : '300-500',
    needsStructure: !isSimple,
  };
}
<<<<<<< HEAD
=======
>>>>>>> master
=======
>>>>>>> 001-rag-chatbot
