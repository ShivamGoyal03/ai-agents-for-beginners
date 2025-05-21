<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9320dd53c82869fd44935d1581eaf7bb",
  "translation_date": "2025-05-21T08:13:02+00:00",
  "source_file": "11-mcp/README.md",
  "language_code": "de"
}
-->
# Lesson 11: Model Context Protocol (MCP) Integration

## Lernziele
- Verstehen, was MCP ist und welche Rolle es bei der Entwicklung von KI-Agenten spielt
- Einrichten und Konfigurieren eines MCP-Servers für die GitHub-Integration
- Aufbau eines Multi-Agenten-Systems mit MCP-Tools
- Umsetzung von RAG (Retrieval Augmented Generation) mit Azure Cognitive Search

## Voraussetzungen
- Python 3.8+
- Node.js 14+
- Azure-Abonnement
- GitHub-Konto
- Grundkenntnisse im Semantic Kernel

## Einrichtung

1. **Umgebung einrichten**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Azure-Dienste konfigurieren**
   - Erstellen einer Azure Cognitive Search-Ressource
   - Einrichten des Azure OpenAI-Dienstes
   - Konfigurieren der Umgebungsvariablen in `.env`

3. **MCP-Server einrichten**
   ```bash
   npm install -g @modelcontextprotocol/server-github
   ```

## Projektstruktur

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

## Kernkomponenten

### 1. Multi-Agenten-System
- GitHub Agent: Analyse von Repositories
- Hackathon Agent: Projektempfehlungen
- Events Agent: Vorschläge für Tech-Events

### 2. Azure-Integration
- Cognitive Search zur Indexierung von Events
- Azure OpenAI für Agentenintelligenz
- Umsetzung des RAG-Musters

### 3. MCP-Tools
- Analyse von GitHub-Repositories
- Code-Inspektion
- Extraktion von Metadaten

## Code-Durchgang

Das Beispiel zeigt:
1. Integration des MCP-Servers
2. Orchestrierung mehrerer Agenten
3. Integration von Azure Cognitive Search
4. Umsetzung des RAG-Musters

Wichtige Funktionen:
- Echtzeit-Analyse von GitHub-Repositories
- Intelligente Projektempfehlungen
- Event-Abgleich mittels Azure Search
- Streaming-Antworten mit Chainlit

## Ausführen des Beispiels

1. Starte den MCP-Server:
   ```bash
   npx @modelcontextprotocol/server-github
   ```

2. Starte die Anwendung:
   ```bash
   chainlit run app.py -w
   ```

3. Teste die Integration:
   ```
   Example query: "Analyze repositories for username: <github_username>"
   ```

## Fehlerbehebung

Häufige Probleme und Lösungen:
1. MCP-Verbindungsprobleme
   - Prüfe, ob der Server läuft
   - Überprüfe die Portverfügbarkeit
   - Bestätige die GitHub-Tokens

2. Azure Search-Probleme
   - Verifiziere die Verbindungsstrings
   - Prüfe, ob der Index vorhanden ist
   - Bestätige den Dokumentenupload

## Nächste Schritte
- Weitere MCP-Tools erkunden
- Eigene Agenten implementieren
- RAG-Funktionalitäten erweitern
- Weitere Event-Quellen hinzufügen

## Ressourcen
- [MCP for Beginners](https://aka.ms/mcp-for-beginners)  
- [MCP Documentation](https://github.com/microsoft/semantic-kernel/tree/main/python/semantic-kernel/semantic_kernel/connectors/mcp)
- [Azure Cognitive Search Docs](https://learn.microsoft.com/azure/search/)
- [Semantic Kernel Guides](https://learn.microsoft.com/semantic-kernel/)

**Haftungsausschluss**:  
Dieses Dokument wurde mit dem KI-Übersetzungsdienst [Co-op Translator](https://github.com/Azure/co-op-translator) übersetzt. Obwohl wir auf Genauigkeit achten, beachten Sie bitte, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner Originalsprache gilt als maßgebliche Quelle. Für wichtige Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die durch die Nutzung dieser Übersetzung entstehen.