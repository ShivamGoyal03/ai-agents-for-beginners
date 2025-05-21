<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9320dd53c82869fd44935d1581eaf7bb",
  "translation_date": "2025-05-21T08:12:21+00:00",
  "source_file": "11-mcp/README.md",
  "language_code": "pt"
}
-->
# Aula 11: Integração do Protocolo de Contexto do Modelo (MCP)

## Objetivos de Aprendizagem
- Compreender o que é o MCP e seu papel no desenvolvimento de agentes de IA
- Configurar e ajustar um servidor MCP para integração com o GitHub
- Construir um sistema multiagente usando as ferramentas MCP
- Implementar RAG (Retrieval Augmented Generation) com Azure Cognitive Search

## Pré-requisitos
- Python 3.8+
- Node.js 14+
- Assinatura Azure
- Conta GitHub
- Conhecimento básico de Semantic Kernel

## Instruções de Configuração

1. **Configuração do Ambiente**  
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configurar Serviços Azure**  
   - Criar um recurso Azure Cognitive Search  
   - Configurar o serviço Azure OpenAI  
   - Ajustar variáveis de ambiente no arquivo `.env`

3. **Configuração do Servidor MCP**  
   ```bash
   npm install -g @modelcontextprotocol/server-github
   ```

## Estrutura do Projeto

```
11-mcp/
├── code_samples/
│   └── github-mcp/
│       ├── app.py              # Main application
│       ├── event-descriptions.md  # Event data
│       └── MCP_SETUP.md        # Setup guide
├── README.md
└── requirements.txt
```

## Componentes Principais

### 1. Sistema Multiagente
- Agente GitHub: Análise de repositórios  
- Agente Hackathon: Recomendações de projetos  
- Agente Eventos: Sugestões de eventos tecnológicos  

### 2. Integração com Azure
- Cognitive Search para indexação de eventos  
- Azure OpenAI para inteligência dos agentes  
- Implementação do padrão RAG  

### 3. Ferramentas MCP
- Análise de repositórios GitHub  
- Inspeção de código  
- Extração de metadados  

## Análise do Código

O exemplo demonstra:  
1. Integração do servidor MCP  
2. Orquestração multiagente  
3. Integração com Azure Cognitive Search  
4. Implementação do padrão RAG  

Principais funcionalidades:  
- Análise em tempo real de repositórios GitHub  
- Recomendações inteligentes de projetos  
- Correspondência de eventos usando Azure Search  
- Respostas em streaming com Chainlit  

## Executando o Exemplo

1. Inicie o servidor MCP:  
   ```bash
   npx @modelcontextprotocol/server-github
   ```

2. Execute a aplicação:  
   ```bash
   chainlit run app.py -w
   ```

3. Teste a integração:  
   ```
   Example query: "Analyze repositories for username: <github_username>"
   ```

## Solução de Problemas

Problemas comuns e soluções:  
1. Problemas de conexão com MCP  
   - Verifique se o servidor está ativo  
   - Cheque a disponibilidade da porta  
   - Confirme os tokens do GitHub  

2. Problemas com Azure Search  
   - Valide as strings de conexão  
   - Verifique a existência do índice  
   - Confirme o upload dos documentos  

## Próximos Passos
- Explorar mais ferramentas MCP  
- Implementar agentes personalizados  
- Aprimorar as capacidades RAG  
- Adicionar mais fontes de eventos  

## Recursos
- [MCP for Beginners](https://aka.ms/mcp-for-beginners)  
- [MCP Documentation](https://github.com/microsoft/semantic-kernel/tree/main/python/semantic-kernel/semantic_kernel/connectors/mcp)  
- [Azure Cognitive Search Docs](https://learn.microsoft.com/azure/search/)  
- [Semantic Kernel Guides](https://learn.microsoft.com/semantic-kernel/)

**Aviso Legal**:  
Este documento foi traduzido utilizando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos para garantir a precisão, esteja ciente de que traduções automáticas podem conter erros ou imprecisões. O documento original em seu idioma nativo deve ser considerado a fonte autorizada. Para informações críticas, recomenda-se a tradução profissional realizada por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações incorretas decorrentes do uso desta tradução.