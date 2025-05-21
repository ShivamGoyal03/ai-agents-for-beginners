<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9320dd53c82869fd44935d1581eaf7bb",
  "translation_date": "2025-05-21T08:10:53+00:00",
  "source_file": "11-mcp/README.md",
  "language_code": "hk"
}
-->
# Lesson 11: Model Context Protocol (MCP) Integration

## Learning Objectives
- 了解 MCP 是什麼，以及它在 AI 代理開發中的角色
- 設置及配置 MCP 伺服器以整合 GitHub
- 使用 MCP 工具構建多代理系統
- 實現搭配 Azure Cognitive Search 的 RAG（檢索增強生成）

## Prerequisites
- Python 3.8+
- Node.js 14+
- Azure 訂閱
- GitHub 帳戶
- 基本了解 Semantic Kernel

## Setup Instructions

1. **環境設置**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **配置 Azure 服務**
   - 建立 Azure Cognitive Search 資源
   - 設定 Azure OpenAI 服務
   - 在 `.env` 中配置環境變數

3. **MCP 伺服器設置**
   ```bash
   npm install -g @modelcontextprotocol/server-github
   ```

## Project Structure

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

## Core Components

### 1. 多代理系統
- GitHub Agent：倉庫分析
- Hackathon Agent：專案推薦
- Events Agent：科技活動建議

### 2. Azure 整合
- 用於活動索引的 Cognitive Search
- 用於代理智慧的 Azure OpenAI
- RAG 模式實作

### 3. MCP 工具
- GitHub 倉庫分析
- 程式碼檢查
- 元資料擷取

## Code Walkthrough

範例展示：
1. MCP 伺服器整合
2. 多代理協調
3. Azure Cognitive Search 整合
4. RAG 模式實作

主要特色：
- 即時 GitHub 倉庫分析
- 智能專案推薦
- 利用 Azure Search 進行活動匹配
- 透過 Chainlit 實現串流回應

## Running the Sample

1. 啟動 MCP 伺服器：
   ```bash
   npx @modelcontextprotocol/server-github
   ```

2. 啟動應用程式：
   ```bash
   chainlit run app.py -w
   ```

3. 測試整合：
   ```
   Example query: "Analyze repositories for username: <github_username>"
   ```

## Troubleshooting

常見問題及解決方法：
1. MCP 連線問題
   - 確認伺服器是否啟動
   - 檢查埠號是否可用
   - 確認 GitHub 權杖有效

2. Azure Search 問題
   - 驗證連線字串
   - 檢查索引是否存在
   - 確認文件已上傳

## Next Steps
- 探索更多 MCP 工具
- 實作自訂代理
- 強化 RAG 功能
- 新增更多活動來源

## Resources
- [MCP for Beginners](https://aka.ms/mcp-for-beginners)  
- [MCP Documentation](https://github.com/microsoft/semantic-kernel/tree/main/python/semantic-kernel/semantic_kernel/connectors/mcp)
- [Azure Cognitive Search Docs](https://learn.microsoft.com/azure/search/)
- [Semantic Kernel Guides](https://learn.microsoft.com/semantic-kernel/)

**免責聲明**：  
本文件係使用 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 進行翻譯。雖然我哋致力於提供準確嘅翻譯，但請注意自動翻譯可能包含錯誤或不準確嘅地方。原文文件嘅母語版本應被視為權威來源。對於重要資訊，建議採用專業人工翻譯。我哋對因使用本翻譯而引起嘅任何誤解或誤釋概不負責。