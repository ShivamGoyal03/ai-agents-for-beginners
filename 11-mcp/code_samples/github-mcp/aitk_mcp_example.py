"""
MCP with AI Toolkit for Visual Studio Code - Multi-Agent Implementation

This script demonstrates how to set up and use the Model Context Protocol (MCP)
with AI Toolkit for Visual Studio Code. It integrates a multi-agent architecture
with GitHub, Hackathon, and Events recommendation agents.
"""

import os
import json
import asyncio
import logging
import re
from dotenv import load_dotenv
import requests
from semantic_kernel.kernel import Kernel
from semantic_kernel.connectors.mcp import MCPStdioPlugin
from semantic_kernel.functions import kernel_function
from semantic_kernel.contents import ChatHistory, AuthorRole, ChatMessageContent
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.contents.function_call_content import FunctionCallContent
from semantic_kernel.contents.function_result_content import FunctionResultContent
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread, AgentGroupChat
from semantic_kernel.agents.strategies import (
    SequentialSelectionStrategy,
    DefaultTerminationStrategy
)
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex, SimpleField, SearchFieldDataType, SearchableField
from azure.core.credentials import AzureKeyCredential

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class RAGPlugin:
    def __init__(self, search_client):
        self.search_client = search_client

    @kernel_function(name="search_events", description="Searches for relevant events based on a query")
    def search_events(self, query: str) -> str:
        """Retrieves relevant events from Azure Search based on the query."""
        context_strings = []
        try:
            results = self.search_client.search(query, top=5)
            for result in results:
                if 'content' in result:
                    context_strings.append(f"Event: {result['content']}")
        except Exception as e:
            context_strings.append(f"Error searching Azure Search: {str(e)}")
        
        # Optional: Add live API integration
        try:
            api_resp = requests.get(f"https://devpost.com/api/hackathons?search={query}", timeout=5)
            if api_resp.ok:
                data = api_resp.json()
                for event in data.get('hackathons', [])[:3]:
                    context_strings.append(f"Live Event: {event.get('title', 'Unknown')} - {event.get('url', '#')}")
        except Exception as e:
            context_strings.append(f"Error fetching live events: {str(e)}")
            
        if context_strings:
            return "\n\n".join(context_strings)
        else:
            return "No relevant events found."

def flatten(xss):
    """Utility function to flatten nested lists."""
    return [x for xs in xss for x in xs]

async def init_search_client():
    """Initialize Azure AI Search with persistent storage."""
    search_service_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
    search_api_key = os.getenv("AZURE_SEARCH_API_KEY")
    index_name = "event-descriptions"

    if not search_service_endpoint or not search_api_key:
        logger.warning("Azure Search credentials not found in environment variables.")
        return None

    search_client = SearchClient(
        endpoint=search_service_endpoint, 
        index_name=index_name,
        credential=AzureKeyCredential(search_api_key)
    )

    index_client = SearchIndexClient(
        endpoint=search_service_endpoint,
        credential=AzureKeyCredential(search_api_key)
    )

    # Define the index schema
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String)
    ]

    index = SearchIndex(name=index_name, fields=fields)

    # Check if index already exists if not, create it
    try:
        existing_index = index_client.get_index(index_name)
        logger.info(f"Index '{index_name}' already exists, using the existing index.")
    except Exception as e:
        # Create the index if it doesn't exist
        logger.info(f"Creating new index '{index_name}'...")
        index_client.create_index(index)

    # Always read event descriptions from markdown file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    event_descriptions_path = os.path.join(current_dir, "event-descriptions.md")

    try:
        with open(event_descriptions_path, "r", encoding='utf-8') as f:
            markdown_content = f.read()
    except FileNotFoundError:
        logger.warning(f"Could not find {event_descriptions_path}")
        markdown_content = ""

    # Split the markdown content into individual event descriptions
    event_descriptions = markdown_content.split("---")  # You can change the delimiter

    # Create documents for Azure Search
    documents = []
    for i, description in enumerate(event_descriptions):
        description = description.strip()  # Remove leading/trailing whitespace
        if description:  # Avoid empty descriptions
            documents.append({"id": str(i + 1), "content": description})

    # Add documents to the index (only if we have documents)
    if documents:
        # Delete existing documents first to avoid duplicates
        try:
            search_client.delete_documents(documents=[{"id": doc["id"]} for doc in documents])
            logger.info("Cleared existing documents")
        except Exception as e:
            logger.warning(f"Failed to clear existing documents: {str(e)}")
        
        # Upload new documents
        search_client.upload_documents(documents)
        logger.info(f"Uploaded {len(documents)} documents to index")

    return search_client

async def setup_github_mcp_plugin():
    """Set up the GitHub MCP plugin for use with AI Toolkit."""
    try:
        logger.info("Initializing GitHub MCP plugin...")
        github_plugin = MCPStdioPlugin(
            name="GitHub",
            description="GitHub Plugin",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"]
        )
        
        # Connect to the GitHub MCP server
        await github_plugin.connect()
        logger.info("GitHub MCP server connection established")
        logger.info("✓ MCP Connection Status: Active")
        logger.info("✓ Plugin Status: Loaded")
        logger.info("✓ Tools: Available")
        
        return github_plugin
    except Exception as e:
        logger.error(f"❌ Error setting up GitHub MCP plugin: {str(e)}")
        logger.error("Check that the GitHub MCP server is running and available.")
        return None

async def call_github_tool(github_plugin, tool_name, tool_input):
    """Call a GitHub MCP tool and return the result."""
    try:
        result = await github_plugin.call_tool(tool_name, tool_input)
        return result
    except Exception as e:
        logger.error(f"❌ Error calling GitHub tool {tool_name}: {str(e)}")
        return f"Error: {str(e)}"

def route_user_input(user_input: str):
    """
    Analyze user input and return a list of agent names to invoke.
    Returns: list of agent names (e.g., ["GitHubAgent", "HackathonAgent", "EventsAgent"])
    """
    user_input_lower = user_input.lower()
    agents = []
    # Example patterns (expand as needed)
    if re.search(r"github|repo|repository|commit|pull request", user_input_lower):
        agents.append("GitHubAgent")
    if re.search(r"hackathon|project idea|competition|challenge|win", user_input_lower):
        agents.append("HackathonAgent")
    if re.search(r"event|conference|meetup|workshop|webinar", user_input_lower):
        agents.append("EventsAgent")
    if not agents:
        agents = ["GitHubAgent", "HackathonAgent", "EventsAgent"]
    return agents

async def setup_agent_architecture(search_client):
    """Set up the multi-agent architecture for AI Toolkit."""
    # Create kernel
    kernel = Kernel()
    
    # Define service ID
    service_id = "agent"
    
    # Add AzureChatCompletion service to kernel
    kernel.add_service(AzureChatCompletion(service_id=service_id))
    
    # Configure execution settings
    settings = kernel.get_prompt_execution_settings_from_service_id(service_id=service_id)
    settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
    
    # Create a RAG plugin if search client is available
    rag_plugin = None
    if search_client:
        rag_plugin = RAGPlugin(search_client)
        kernel.add_plugin(rag_plugin, plugin_name="RAG")
        logger.info("RAG plugin initialized successfully")
    
    # Set up the GitHub MCP plugin
    github_plugin = await setup_github_mcp_plugin()
    if not github_plugin:
        logger.error("Failed to set up GitHub MCP plugin. Agent capabilities will be limited.")
    else:
        kernel.add_plugin(github_plugin)
    
    # Define agent instructions
    GITHUB_INSTRUCTIONS = """
You are an expert on GitHub repositories. When answering questions, you **must** use the provided GitHub username to find specific information about that user's repositories, including:

*   Who created the repositories
*   The programming languages used
*   Information found in files and README.md files within those repositories
*   Provide links to each repository referenced in your answers

**Important:** Never perform general searches for repositories. Always use the given GitHub username to find the relevant information. If a GitHub username is not provided, state that you need a username to proceed.
    """

    HACKATHON_AGENT = """
You are an AI Agent Hackathon Strategist specializing in recommending winning project ideas.

Your task:
1. Analyze the GitHub activity of users to understand their technical skills
2. Suggest creative AI Agent projects tailored to their expertise. 
3. Focus on projects that align with Microsoft's AI Agent Hackathon prize categories

When making recommendations:
- Base your ideas strictly on the user's GitHub repositories, languages, and tools
- Give suggestions on tools, languages and frameworks to use to build it. 
- Provide detailed project descriptions including architecture and implementation approach
- Explain why the project has potential to win in specific prize categories
- Highlight technical feasibility given the user's demonstrated skills by referencing the specific repositories or languages used.

Formatting your response:
- Provide a clear and structured response that includes:
    - Suggested Project Name
    - Project Description 
    - Potential languages and tools to use
    - Link to each relevant GitHub repository you based your recommendation on

Hackathon prize categories:
- Best Overall Agent ($20,000)
- Best Agent in Python ($5,000)
- Best Agent in C# ($5,000)
- Best Agent in Java ($5,000)
- Best Agent in JavaScript/TypeScript ($5,000)
- Best Copilot Agent using Microsoft Copilot Studio or Microsoft 365 Agents SDK ($5,000)
- Best Azure AI Agent Service Usage ($5,000)
    """

    EVENTS_AGENT = """
You are an Event Recommendation Agent specializing in suggesting relevant tech events.

Your task:
1. Review the project idea recommended by the Hackathon Agent
2. Use the search_events function to find relevant events based on the technologies mentioned.
3. NEVER suggest and event that the where there is not a relevant technology that the user has used.
3. ONLY recommend events that were returned by the search_events functionf

When making recommendations:
- IMPORTANT: You must first call the search_events function with appropriate technology keywords from the project
- Only recommend events that were explicitly returned by the search_events function
- Do not make up or suggest events that weren't in the search results
- Construct search queries using specific technologies mentioned (e.g., "Python AI workshop" or "JavaScript hackathon")
- Try multiple search queries if needed to find the most relevant events

For each recommended event:
- Only include events found in the search_events results
- Explain the direct connection between the event and the specific project requirements
- Highlight relevant workshops, sessions, or networking opportunities

Formatting your response:
- Start with "Based on the hackathon project idea, here are relevant events that I found:"
- Only list events that were returned by the search_events function
- For each event, include the exact event details as returned by search_events
- Explain specifically how each event relates to the project technologies

If no relevant events are found, acknowledge this and suggest trying different search terms instead of making up events.
    """
    
    # Create agents
    github_agent = None
    hackathon_agent = None
    events_agent = None
    
    if github_plugin:
        github_agent = ChatCompletionAgent(
            service=AzureChatCompletion(),
            name="GitHubAgent",
            instructions=GITHUB_INSTRUCTIONS,
            plugins=[github_plugin]
        )
    
    hackathon_agent = ChatCompletionAgent(
        service=AzureChatCompletion(),
        name="HackathonAgent",
        instructions=HACKATHON_AGENT
    )
    
    if rag_plugin:
        events_agent = ChatCompletionAgent(
            service=AzureChatCompletion(),
            name="EventsAgent",
            instructions=EVENTS_AGENT,
            plugins=[rag_plugin]
        )
    
    # Create agent group chat if all agents are available
    agent_group_chat = None
    if github_agent and hackathon_agent and events_agent:
        agent_group_chat = AgentGroupChat(
            agents=[github_agent, hackathon_agent, events_agent],
            selection_strategy=SequentialSelectionStrategy(initial_agent=github_agent),
            termination_strategy=DefaultTerminationStrategy(maximum_iterations=3)
        )
        logger.info("Agent group chat initialized successfully")
    
    # Create a new chat history
    chat_history = ChatHistory()
    
    return {
        "kernel": kernel,
        "settings": settings,
        "github_plugin": github_plugin,
        "rag_plugin": rag_plugin,
        "chat_history": chat_history,
        "agent_group_chat": agent_group_chat,
        "github_agent": github_agent,
        "hackathon_agent": hackathon_agent,
        "events_agent": events_agent
    }

async def interactive_agent_chat(agent_context):
    """Interactive console chat with the multi-agent system."""
    kernel = agent_context["kernel"]
    chat_history = agent_context["chat_history"]
    github_agent = agent_context["github_agent"] 
    hackathon_agent = agent_context["hackathon_agent"]
    events_agent = agent_context["events_agent"]
    agent_group_chat = agent_context["agent_group_chat"]
    chat_completion_service = kernel.get_service("agent")
    
    logger.info("\n=== AI Toolkit Multi-Agent Chat ===")
    logger.info("Type 'exit' to quit the chat")
    logger.info("Available agents: " + ", ".join([a.name for a in [github_agent, hackathon_agent, events_agent] if a]))
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            break
        
        # Add user message to chat history
        chat_history.add_user_message(user_input)
        
        # Determine which agents to use based on user input
        agent_names = route_user_input(user_input)
        print(f"Processing with agents: {', '.join(agent_names)}")
        
        # If more than one agent is selected and agent group chat is available, use it
        if len(agent_names) > 1 and agent_group_chat:
            try:
                await agent_group_chat.add_chat_message(user_input)
                print("\nProcessing with agent group chat...")
                agent_responses = []
                
                async for content in agent_group_chat.invoke():
                    agent_name = content.name or "Agent"
                    response = f"\n{agent_name}: {content.content}"
                    agent_responses.append(response)
                    print(response)
                
                full_response = "\n".join(agent_responses)
                chat_history.add_assistant_message(full_response)
            except Exception as e:
                error_msg = f"Error in agent group chat: {str(e)}"
                print(error_msg)
                logger.error(error_msg)
                chat_history.add_assistant_message(f"Error: {str(e)}")
        else:
            # Single agent mode
            try:
                agent_name = agent_names[0]
                print(f"\nProcessing with {agent_name}...")
                
                async for msg in chat_completion_service.get_streaming_chat_message_content(
                    chat_history=chat_history,
                    user_input=user_input,
                    settings=agent_context["settings"],
                    kernel=kernel,
                ):
                    if msg.content:
                        print(msg.content, end="")
                    if isinstance(msg, FunctionCallContent):
                        function_name = msg.function_name
                        print(f"\n\nCalling function: {function_name}")
                    if isinstance(msg, FunctionResultContent):
                        print(f"\nFunction result: {msg.content}")
                
                # Capture the last assistant message from chat history
                last_message = chat_history.messages[-1] if chat_history.messages else None
                if last_message and last_message.role != AuthorRole.USER:
                    print("\n")  # Add a newline for readability
            except Exception as e:
                error_msg = f"Error in single agent mode: {str(e)}"
                print(error_msg)
                logger.error(error_msg)
                chat_history.add_assistant_message(f"Error: {str(e)}")

async def main():
    """Main entrypoint for the AI Toolkit MCP example."""
    logger.info("Setting up AI Toolkit MCP example with multi-agent architecture...")
    
    # Initialize search client
    search_client = await init_search_client()
    if not search_client:
        logger.warning("Failed to initialize search client. RAG functionality will be limited.")
    
    # Set up the agent architecture
    agent_context = await setup_agent_architecture(search_client)
    
    logger.info("\n=== AI Toolkit MCP Setup Complete ===")
    logger.info("You can either:")
    logger.info("1. Use the interactive console chat (starting now)")
    logger.info("2. Use the AI Toolkit playground to interact with the MCP server")
    
    # Start interactive console chat
    try:
        await interactive_agent_chat(agent_context)
    except KeyboardInterrupt:
        logger.info("Chat interrupted...")
    finally:
        # Clean up resources
        if agent_context["github_plugin"]:
            await agent_context["github_plugin"].close()
            logger.info("MCP plugin closed successfully")
        
        logger.info("Goodbye!")

if __name__ == "__main__":
    asyncio.run(main())
