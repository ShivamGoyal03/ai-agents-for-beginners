# Building a Multi-Agent Travel Planning System

## What I'm Trying to Achieve

This project aims to create an orchestrated multi-agent system for comprehensive travel planning. The system uses specialized agents (flight, hotel, restaurant, and excursion) that work together seamlessly to provide a complete travel planning experience. Rather than forcing users to interact with multiple separate assistants, this approach creates a cohesive conversation flow where each agent handles its specialized domain.

## The Architecture

**Title: Orchestrated Multi-Agent Travel Planner**

**Description:** An intelligent travel planning system that coordinates multiple specialized AI agents to provide comprehensive trip planning services. Using Azure AI Agent Service for deployment and Semantic Kernel for orchestration, the system creates a seamless conversation flow across different travel domains.

**Key Components:**
- **Agents**: Specialized agents for flights, hotels, restaurants, and excursions
- **Orchestration**: Semantic Kernel's `AgentGroupChat` for conversation flow management
- **Deployment**: Azure AI Agent Service for agent deployment and management

## Setup Instructions

1. Install the required dependencies
    ```bash
    AZURE_AI_AGENT_PROJECT_CONNECTION_STRING="<your-connection-string>"
    AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME="<your-deployment-name>"
    GITHUB_TOKEN="<your-token>"
    ```

## Current Implementation

The system uses four specialized agents deployed through Azure AI Agent Service:
1. **Flight Agent** - Handles flight searches and bookings
2. **Hotel Agent** - Manages accommodation recommendations and reservations
3. **Restaurant Agent** - Provides dining recommendations based on location and preferences
4. **Excursion Agent** - Suggests and plans activities and sightseeing options

The orchestration is handled by Semantic Kernel's `AgentGroupChat`, which manages the conversation flow between agents using custom selection and termination strategies.

## Conversation Flow

1. **User Input**: The user starts by providing their travel preferences and requirements
2. **Agent Selection**: The system routes the query to the appropriate agent based on the user's input
3. **Agent Interaction**: Each agent handles its specialized task (flight search, hotel booking, etc.)
4. **Handoff**: Agents pass context and relevant information to the next agent in the flow
5. **Task Completion**: The system detects when all tasks are completed and ends the conversation

## Challenges Faced

While the system is functional, several challenges have emerged in the coordination and orchestration of the specialized agents. These include issues with agent selection logic, template rendering errors, and conversation flow management. Addressing these challenges is crucial to creating a seamless and efficient travel planning experience.


## Blockers: Agent Coordination Challenges

Several challenges have emerged:

1. **Agent Selection Logic Issues**:
   - The current selection strategy sometimes fails to properly parse agent names or select the correct next agent
   - The orchestration prompts need fine-tuning to better understand conversation state

2. **Template Rendering Errors**:
   - Experiencing `TemplateRenderException` errors when trying to pass conversation context between agents
   - Variable handling in templates (using `{{$lastmessage}}`) is inconsistent

3. **Agent Response Processing**:
   - Some agent responses are not being correctly formatted or displayed
   - Need better parsing of response types and handling of multi-turn conversations

4. **Conversation Flow Management**:
   - Agents sometimes repeat information or fail to pick up where the previous agent left off
   - Need better context preservation between agent transitions

5. **Known Issues**
   ```python
   AgentChatException: Failed to select agent: Agent Failure - 
   Strategy unable to select next agent
   ```
   - Currently addressing agent selection and mapping
   - Improving task completion detection

## Improvement Direction

To address these issues:
1. Implement a more robust agent selection function with better parsing logic
2. Create more explicit handoff protocols between agents
3. Improve error handling for template rendering
4. Add logging and monitoring to track conversation state
5. Implement user intent detection to better route initial queries

## Desired Outcome

The goal is to create a seamless experience where these specialized agents work together coherently, handling complex travel planning tasks that span multiple domains while maintaining conversation context.

## Contact

If you have experience with multi-agent orchestration using Semantic Kernel or Azure AI Agent Service and can help address these challenges, please reach out to me at [Shivam Goyal](https://www.linkedin.com/in/shivam2003/) on LinkedIn.

<table>
<tr>
    <td align="center"><a href="https://github.com/ShivamGoyal03">
        <img src="https://github.com/ShivamGoyal03.png" width="100px;" alt="Shivam Goyal"/><br />
        <sub><b>Shivam Goyal</b></sub>
    </a><br />
    </td>
</tr></table>

## Future Enhancements

1. **User Context Management**:
   - Implement user context tracking to maintain state across multiple interactions
   - Use context to personalize recommendations and improve conversational flow

2. **Web Interface with Chainlit**:
   - Deploy the multi-agent system as an interactive web application using Chainlit
   - Provide a user-friendly interface with agent responses clearly visualized

3. **Integration with External APIs**:
   - Connect agents to external APIs for real-time data retrieval (e.g., flight availability, hotel bookings)
   - Enhance agent capabilities with live data feeds and dynamic recommendations


## Acknowledgements

Special thanks to the Azure AI Agent Service and [Semantic Kernel](https://github.com/microsoft/semantic-kernel) teams for their support and guidance in building this multi-agent travel planning system.

- [Semantic Kernel Cookbook](https://sphenry.github.io/sk-cookbook-agents/)