# Using AI Toolkit for Model Context Protocol (MCP) Testing

## Introduction

While Chainlit offers an excellent chat interface for testing MCP servers, Microsoft's AI Toolkit for Visual Studio Code provides a more integrated and beginner-friendly alternative. AI Toolkit allows you to build, test, and debug MCP clients and servers directly within your development environment, providing a seamless experience for AI application development.

This guide walks you through setting up and using AI Toolkit as an alternative to Chainlit for testing your Model Context Protocol (MCP) implementations.

## Why Use AI Toolkit for MCP Testing?

- **Integrated Development Experience**: Test MCP directly within VS Code without needing to set up separate web interfaces
- **Simplified Model Management**: Access to a wide variety of AI models through a unified catalog
- **Streamlined Testing**: Quick testing of prompts and responses with the built-in playground
- **Attachment Support**: Easily test multi-modal capabilities with file attachments
- **Advanced Evaluation Tools**: Evaluate model performance with built-in metrics and batch processing
- **No Extra Dependencies**: No need for additional web frameworks or Python packages beyond your MCP implementation

## Prerequisites

- [Visual Studio Code](https://code.visualstudio.com/) (latest version recommended)
- [Node.js](https://nodejs.org/) (version 14 or higher)
- [AI Toolkit for VS Code](https://marketplace.visualstudio.com/items?itemName=ms-windows-ai-studio.windows-ai-studio) extension installed
- GitHub Personal Access Token (PAT) for using GitHub Models

## Setup Instructions

### 1. Install AI Toolkit Extension

1. Open Visual Studio Code
2. Navigate to Extensions (Ctrl+Shift+X or Cmd+Shift+X on macOS)
3. Search for "AI Toolkit"
4. Install "AI Toolkit for Visual Studio Code" by Microsoft

### 2. Configure MCP Server

1. Create a file named `mcp-server.js` in your project directory with the following content:

```javascript
// Import required modules for your MCP server
// This example shows a simple setup for creating an MCP server with GitHub integration

// Start the MCP server when this script is executed
console.log("Starting GitHub MCP server...");
// Your server setup code would go here
```

### 3. Configure Environment Variables

1. Create or update your `.env` file with your GitHub Personal Access Token:

```
GITHUB_PERSONAL_ACCESS_TOKEN=your_github_pat_here
```

## Using AI Toolkit with MCP

### Method 1: Direct MCP Server Launch

1. Open a terminal in VS Code (Terminal > New Terminal)
2. Start the MCP server using:

```bash
npx -y @modelcontextprotocol/server-github --env GITHUB_PERSONAL_ACCESS_TOKEN=your_github_pat_here
```

3. Open AI Toolkit from the activity bar (Left side)
4. Click on "Playground" in the AI Toolkit view
5. Your MCP server will now be available to integrate with model interactions

### Method 2: Launch Through AI Toolkit Interface

1. Open the AI Toolkit view in VS Code
2. Click on "Add Tool" or "Connect to External Tool"
3. Select "Connect to MCP Server"
4. Enter the following details:
   - Name: GitHub MCP
   - Command: npx -y @modelcontextprotocol/server-github
   - Arguments: --env GITHUB_PERSONAL_ACCESS_TOKEN=your_github_pat_here
5. Click "Connect" to start the MCP server

## Testing Your MCP Implementation

Once your MCP server is running, you can test it using the AI Toolkit playground:

1. Open the AI Toolkit playground
2. Type a query like "Show me repositories for username: [GitHub Username]"
3. Send the message and observe the response
4. The playground will show you the full interaction, including:
   - The initial model prompt
   - Tool calls made by the model to your MCP server
   - Responses from your MCP server
   - The final assistant response

## Debugging MCP with AI Toolkit

AI Toolkit provides several features to help debug your MCP implementation:

1. **Response Inspector**: View the raw JSON responses from your MCP server
2. **Tool Call Tracking**: See which tools were called and with what parameters
3. **Error Highlighting**: Quickly identify issues in your MCP server responses
4. **Console Integration**: View server logs directly in the VS Code terminal

## Advanced Features

### Testing with Multiple Models

AI Toolkit allows you to test your MCP server with different models:

1. In the playground, click on the model selector dropdown
2. Choose from available models (GPT-4o, Claude, Llama, etc.)
3. Test the same MCP interactions across different models to ensure compatibility

### Batch Testing

For comprehensive testing of your MCP server:

1. Create a test dataset with various GitHub query scenarios
2. Use AI Toolkit's Bulk Run feature to execute all test cases
3. Compare results across different models or server configurations

### Evaluating MCP Performance

AI Toolkit includes evaluation tools to assess your MCP server's performance:

1. Navigate to the Evaluation section in AI Toolkit
2. Select your test dataset and evaluation metrics (accuracy, latency, etc.)
3. Run the evaluation to get performance insights

## Additional Resources

- [AI Toolkit for Visual Studio Code Documentation](https://code.visualstudio.com/docs/intelligentapps/overview)
- [Model Context Protocol Official Documentation](https://modelcontextprotocol.io/docs/)
- [GitHub Models Documentation](https://aka.ms/ai-agents-beginners/github-models)

## Troubleshooting

### Common Issues and Solutions

1. **Connection Errors**:
   - Verify your GitHub token is correctly set in the environment variables
   - Check if the MCP server port is available (default is 3000)

2. **Tool Not Found**:
   - Ensure your MCP server is properly registering tools
   - Check server logs for registration errors

3. **Authentication Issues**:
   - Confirm your GitHub PAT has the correct scopes (public_repo at minimum)
   - Try regenerating your token if permissions have changed

4. **Response Timeout**:
   - MCP operations may take time for complex queries
   - Increase timeout settings in your configuration
