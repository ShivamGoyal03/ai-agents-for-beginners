<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9320dd53c82869fd44935d1581eaf7bb",
  "translation_date": "2025-05-21T08:11:12+00:00",
  "source_file": "11-mcp/README.md",
  "language_code": "fr"
}
-->
# Leçon 11 : Intégration du protocole Model Context (MCP)

## Objectifs d’apprentissage
- Comprendre ce qu’est le MCP et son rôle dans le développement d’agents IA
- Installer et configurer un serveur MCP pour l’intégration GitHub
- Construire un système multi-agents avec les outils MCP
- Implémenter RAG (Retrieval Augmented Generation) avec Azure Cognitive Search

## Prérequis
- Python 3.8+
- Node.js 14+
- Abonnement Azure
- Compte GitHub
- Connaissances de base en Semantic Kernel

## Instructions d’installation

1. **Configuration de l’environnement**  
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configurer les services Azure**  
   - Créer une ressource Azure Cognitive Search  
   - Mettre en place le service Azure OpenAI  
   - Configurer les variables d’environnement dans `.env`

3. **Installation du serveur MCP**  
   ```bash
   npm install -g @modelcontextprotocol/server-github
   ```

## Structure du projet

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

## Composants principaux

### 1. Système multi-agents
- Agent GitHub : analyse des dépôts  
- Agent Hackathon : recommandations de projets  
- Agent Événements : suggestions d’événements tech

### 2. Intégration Azure
- Cognitive Search pour l’indexation des événements  
- Azure OpenAI pour l’intelligence des agents  
- Implémentation du modèle RAG

### 3. Outils MCP
- Analyse des dépôts GitHub  
- Inspection du code  
- Extraction des métadonnées

## Parcours du code

L’exemple illustre :  
1. L’intégration du serveur MCP  
2. L’orchestration multi-agents  
3. L’intégration d’Azure Cognitive Search  
4. La mise en œuvre du modèle RAG

Fonctionnalités clés :  
- Analyse en temps réel des dépôts GitHub  
- Recommandations de projets intelligentes  
- Correspondance d’événements via Azure Search  
- Réponses en streaming avec Chainlit

## Exécution de l’exemple

1. Démarrer le serveur MCP :  
   ```bash
   npx @modelcontextprotocol/server-github
   ```

2. Lancer l’application :  
   ```bash
   chainlit run app.py -w
   ```

3. Tester l’intégration :  
   ```
   Example query: "Analyze repositories for username: <github_username>"
   ```

## Dépannage

Problèmes courants et solutions :  
1. Problèmes de connexion MCP  
   - Vérifier que le serveur est bien lancé  
   - Contrôler la disponibilité du port  
   - Confirmer les tokens GitHub

2. Problèmes Azure Search  
   - Valider les chaînes de connexion  
   - Vérifier l’existence de l’index  
   - Confirmer le téléchargement des documents

## Étapes suivantes
- Explorer d’autres outils MCP  
- Créer des agents personnalisés  
- Améliorer les capacités RAG  
- Ajouter plus de sources d’événements

## Ressources
- [MCP for Beginners](https://aka.ms/mcp-for-beginners)  
- [MCP Documentation](https://github.com/microsoft/semantic-kernel/tree/main/python/semantic-kernel/semantic_kernel/connectors/mcp)  
- [Azure Cognitive Search Docs](https://learn.microsoft.com/azure/search/)  
- [Semantic Kernel Guides](https://learn.microsoft.com/semantic-kernel/)

**Avertissement** :  
Ce document a été traduit à l'aide du service de traduction automatique [Co-op Translator](https://github.com/Azure/co-op-translator). Bien que nous nous efforcions d'assurer l'exactitude, veuillez noter que les traductions automatiques peuvent contenir des erreurs ou des inexactitudes. Le document original dans sa langue d'origine doit être considéré comme la source faisant foi. Pour les informations critiques, une traduction professionnelle réalisée par un humain est recommandée. Nous déclinons toute responsabilité en cas de malentendus ou de mauvaises interprétations résultant de l'utilisation de cette traduction.