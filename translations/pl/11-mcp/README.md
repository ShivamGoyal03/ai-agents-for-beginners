<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9320dd53c82869fd44935d1581eaf7bb",
  "translation_date": "2025-05-21T08:13:46+00:00",
  "source_file": "11-mcp/README.md",
  "language_code": "pl"
}
-->
# Lesson 11: Model Context Protocol (MCP) Integration

## Cele nauki
- Zrozumieć, czym jest MCP i jaka jest jego rola w tworzeniu agentów AI
- Skonfigurować i uruchomić serwer MCP do integracji z GitHub
- Zbudować system wieloagentowy przy użyciu narzędzi MCP
- Wdrożyć RAG (Retrieval Augmented Generation) z wykorzystaniem Azure Cognitive Search

## Wymagania wstępne
- Python 3.8+
- Node.js 14+
- Subskrypcja Azure
- Konto GitHub
- Podstawowa znajomość Semantic Kernel

## Instrukcje konfiguracji

1. **Konfiguracja środowiska**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Konfiguracja usług Azure**
   - Utwórz zasób Azure Cognitive Search
   - Skonfiguruj usługę Azure OpenAI
   - Ustaw zmienne środowiskowe w pliku `.env`

3. **Uruchomienie serwera MCP**
   ```bash
   npm install -g @modelcontextprotocol/server-github
   ```

## Struktura projektu

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

## Główne komponenty

### 1. System wieloagentowy
- GitHub Agent: analiza repozytorium
- Hackathon Agent: rekomendacje projektów
- Events Agent: sugestie wydarzeń technologicznych

### 2. Integracja z Azure
- Cognitive Search do indeksowania wydarzeń
- Azure OpenAI dla inteligencji agentów
- Implementacja wzorca RAG

### 3. Narzędzia MCP
- Analiza repozytoriów GitHub
- Inspekcja kodu
- Ekstrakcja metadanych

## Omówienie kodu

Przykład pokazuje:
1. Integrację serwera MCP
2. Orkiestrację systemu wieloagentowego
3. Integrację z Azure Cognitive Search
4. Implementację wzorca RAG

Kluczowe funkcje:
- Analiza repozytoriów GitHub w czasie rzeczywistym
- Inteligentne rekomendacje projektów
- Dopasowywanie wydarzeń z użyciem Azure Search
- Odpowiedzi strumieniowe z Chainlit

## Uruchamianie przykładu

1. Uruchom serwer MCP:
   ```bash
   npx @modelcontextprotocol/server-github
   ```

2. Włącz aplikację:
   ```bash
   chainlit run app.py -w
   ```

3. Przetestuj integrację:
   ```
   Example query: "Analyze repositories for username: <github_username>"
   ```

## Rozwiązywanie problemów

Typowe problemy i rozwiązania:
1. Problemy z połączeniem MCP
   - Sprawdź, czy serwer działa
   - Zweryfikuj dostępność portu
   - Potwierdź poprawność tokenów GitHub

2. Problemy z Azure Search
   - Sprawdź poprawność connection stringów
   - Zweryfikuj istnienie indeksu
   - Upewnij się, że dokumenty zostały przesłane

## Kolejne kroki
- Poznaj dodatkowe narzędzia MCP
- Stwórz własnych agentów
- Rozbuduj możliwości RAG
- Dodaj więcej źródeł wydarzeń

## Zasoby
- [MCP for Beginners](https://aka.ms/mcp-for-beginners)  
- [MCP Documentation](https://github.com/microsoft/semantic-kernel/tree/main/python/semantic-kernel/semantic_kernel/connectors/mcp)
- [Azure Cognitive Search Docs](https://learn.microsoft.com/azure/search/)
- [Semantic Kernel Guides](https://learn.microsoft.com/semantic-kernel/)

**Zastrzeżenie**:  
Niniejszy dokument został przetłumaczony za pomocą usługi tłumaczeń AI [Co-op Translator](https://github.com/Azure/co-op-translator). Mimo że dokładamy starań, aby tłumaczenie było jak najdokładniejsze, prosimy mieć na uwadze, że automatyczne tłumaczenia mogą zawierać błędy lub niedokładności. Oryginalny dokument w języku źródłowym należy traktować jako źródło autorytatywne. W przypadku informacji krytycznych zaleca się skorzystanie z profesjonalnego tłumaczenia wykonanego przez człowieka. Nie ponosimy odpowiedzialności za jakiekolwiek nieporozumienia lub błędne interpretacje wynikające z użycia tego tłumaczenia.