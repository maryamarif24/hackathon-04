#!/usr/bin/env node

/**
 * Physical AI & Humanoid Robotics Textbook MCP Server
 *
 * This MCP server provides AI assistants with access to query the textbook content.
 * It exposes tools for searching chapters, getting chapter summaries, and more.
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';

// Textbook content database
const TEXTBOOK_CONTENT = {
  chapters: [
    {
      id: 1,
      title: "Introduction to Physical AI",
      sections: [
        {
          id: "1.1",
          title: "What is Physical AI?",
          content: "Physical AI refers to artificial intelligence systems that interact directly with the physical world through robotic platforms. Unlike traditional AI that operates purely in software, Physical AI combines perception, cognition, and action."
        },
        {
          id: "1.2",
          title: "Key Applications",
          content: "Physical AI is critical for humanoid robots, autonomous vehicles, industrial automation, warehouse robotics, and healthcare assistance robots."
        },
        {
          id: "1.3",
          title: "Core Components",
          content: "The three pillars of Physical AI are: Perception (sensors like cameras, LiDAR, force sensors), Cognition (AI models for decision-making), and Action (actuators and motors)."
        }
      ]
    },
    {
      id: 2,
      title: "Basics of Humanoid Robotics",
      sections: [
        {
          id: "2.1",
          title: "Mechanical Design",
          content: "Humanoid robots use joints, actuators, and structural elements that mimic human anatomy, enabling bipedal locomotion and dexterous manipulation."
        },
        {
          id: "2.2",
          title: "Sensors and Perception",
          content: "Vision systems, tactile sensors, IMUs for balance, and proprioceptive sensors provide robots with understanding of their environment and body state."
        },
        {
          id: "2.3",
          title: "Modern Examples",
          content: "Tesla Optimus, Boston Dynamics Atlas, and Figure 01 demonstrate state-of-the-art humanoid robotics with advanced mobility and dexterity."
        }
      ]
    },
    {
      id: 3,
      title: "ROS 2 Fundamentals",
      sections: [
        {
          id: "3.1",
          title: "ROS 2 Architecture",
          content: "ROS 2 is built on DDS (Data Distribution Service) middleware, providing nodes, topics, services, and actions for distributed robot systems."
        },
        {
          id: "3.2",
          title: "Communication Patterns",
          content: "Topics for pub/sub messaging, Services for request/response, Actions for long-running tasks with feedback."
        },
        {
          id: "3.3",
          title: "Tools and Ecosystem",
          content: "RViz for visualization, Gazebo for simulation, rqt for debugging, and Nav2 for navigation."
        }
      ]
    },
    {
      id: 4,
      title: "Digital Twin Simulation",
      sections: [
        {
          id: "4.1",
          title: "Gazebo Simulator",
          content: "Gazebo provides physics simulation, sensor simulation, and robot modeling for testing in virtual environments."
        },
        {
          id: "4.2",
          title: "NVIDIA Isaac Sim",
          content: "Isaac Sim offers photorealistic rendering, GPU-accelerated physics, and domain randomization for training robust AI models."
        },
        {
          id: "4.3",
          title: "Sim-to-Real Transfer",
          content: "Techniques for transferring models trained in simulation to real robots, including domain randomization and reality gap bridging."
        }
      ]
    },
    {
      id: 5,
      title: "Vision-Language-Action Systems",
      sections: [
        {
          id: "5.1",
          title: "VLA Architecture",
          content: "VLA systems combine vision (cameras), language (NLP), and action (robot control) into unified models like RT-2 and PaLM-E."
        },
        {
          id: "5.2",
          title: "Training VLA Models",
          content: "Large-scale datasets of robot interactions, language instructions, and visual observations are used to train VLA models."
        },
        {
          id: "5.3",
          title: "Real-World Deployment",
          content: "VLA models enable robots to understand commands like 'pick up the red cup' and execute corresponding actions with minimal task-specific training."
        }
      ]
    },
    {
      id: 6,
      title: "Capstone: Simple AI-Robot Pipeline",
      sections: [
        {
          id: "6.1",
          title: "System Design",
          content: "End-to-end pipeline from perception to action, integrating vision, planning, and control."
        },
        {
          id: "6.2",
          title: "Implementation",
          content: "Building a simple pick-and-place robot using ROS 2, OpenCV for vision, and MoveIt for motion planning."
        },
        {
          id: "6.3",
          title: "Best Practices",
          content: "Testing strategies, error handling, logging, and deployment considerations for production robot systems."
        }
      ]
    }
  ]
};

// Create MCP server
const server = new Server(
  {
    name: 'physical-ai-textbook-server',
    version: '2.0.0',
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

// List available tools
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: 'search_textbook',
        description: 'Search the Physical AI textbook for specific topics or keywords',
        inputSchema: {
          type: 'object',
          properties: {
            query: {
              type: 'string',
              description: 'Search query (e.g., "ROS 2", "humanoid robots", "VLA systems")',
            },
          },
          required: ['query'],
        },
      },
      {
        name: 'get_chapter',
        description: 'Get the full content of a specific chapter by ID',
        inputSchema: {
          type: 'object',
          properties: {
            chapter_id: {
              type: 'number',
              description: 'Chapter number (1-6)',
            },
          },
          required: ['chapter_id'],
        },
      },
      {
        name: 'list_chapters',
        description: 'List all available chapters in the textbook',
        inputSchema: {
          type: 'object',
          properties: {},
        },
      },
      {
        name: 'get_section',
        description: 'Get a specific section from a chapter',
        inputSchema: {
          type: 'object',
          properties: {
            chapter_id: {
              type: 'number',
              description: 'Chapter number (1-6)',
            },
            section_id: {
              type: 'string',
              description: 'Section ID (e.g., "1.1", "3.2")',
            },
          },
          required: ['chapter_id', 'section_id'],
        },
      },
    ],
  };
});

// Handle tool calls
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  switch (name) {
    case 'search_textbook': {
      const query = args.query.toLowerCase();
      const results = [];

      TEXTBOOK_CONTENT.chapters.forEach((chapter) => {
        chapter.sections.forEach((section) => {
          if (
            section.title.toLowerCase().includes(query) ||
            section.content.toLowerCase().includes(query)
          ) {
            results.push({
              chapter_id: chapter.id,
              chapter_title: chapter.title,
              section_id: section.id,
              section_title: section.title,
              content: section.content,
            });
          }
        });
      });

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(
              {
                query,
                results_count: results.length,
                results,
              },
              null,
              2
            ),
          },
        ],
      };
    }

    case 'list_chapters': {
      const chapters = TEXTBOOK_CONTENT.chapters.map((ch) => ({
        id: ch.id,
        title: ch.title,
        sections_count: ch.sections.length,
      }));

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify({ chapters }, null, 2),
          },
        ],
      };
    }

    case 'get_chapter': {
      const chapter = TEXTBOOK_CONTENT.chapters.find(
        (ch) => ch.id === args.chapter_id
      );

      if (!chapter) {
        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify({
                error: `Chapter ${args.chapter_id} not found`,
              }),
            },
          ],
        };
      }

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(chapter, null, 2),
          },
        ],
      };
    }

    case 'get_section': {
      const chapter = TEXTBOOK_CONTENT.chapters.find(
        (ch) => ch.id === args.chapter_id
      );

      if (!chapter) {
        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify({
                error: `Chapter ${args.chapter_id} not found`,
              }),
            },
          ],
        };
      }

      const section = chapter.sections.find((s) => s.id === args.section_id);

      if (!section) {
        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify({
                error: `Section ${args.section_id} not found in chapter ${args.chapter_id}`,
              }),
            },
          ],
        };
      }

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(
              {
                chapter_id: chapter.id,
                chapter_title: chapter.title,
                section,
              },
              null,
              2
            ),
          },
        ],
      };
    }

    default:
      throw new Error(`Unknown tool: ${name}`);
  }
});

// Start server
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error('Physical AI Textbook MCP Server running on stdio');
}

main().catch((error) => {
  console.error('Server error:', error);
  process.exit(1);
});
