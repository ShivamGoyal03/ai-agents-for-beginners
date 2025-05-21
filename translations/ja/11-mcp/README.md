<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9320dd53c82869fd44935d1581eaf7bb",
  "translation_date": "2025-05-21T08:11:34+00:00",
  "source_file": "11-mcp/README.md",
  "language_code": "ja"
}
-->
# Lesson 11: Model Context Protocol (MCP) Integration

## Learning Objectives
- MCPとは何か、そしてAIエージェント開発における役割を理解する
- GitHub連携のためのMCPサーバーをセットアップ・構成する
- MCPツールを使ったマルチエージェントシステムを構築する
- Azure Cognitive Searchを用いたRAG（Retrieval Augmented Generation）を実装する

## Prerequisites
- Python 3.8以上
- Node.js 14以上
- Azureサブスクリプション
- GitHubアカウント
- Semantic Kernelの基本知識

## Setup Instructions

1. **環境設定**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Azureサービスの構成**
   - Azure Cognitive Searchリソースを作成する
   - Azure OpenAIサービスをセットアップする
   - `.env`に環境変数を設定する

3. **MCPサーバーのセットアップ**
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

### 1. マルチエージェントシステム
- GitHub Agent：リポジトリ解析
- Hackathon Agent：プロジェクト推薦
- Events Agent：技術イベントの提案

### 2. Azure連携
- イベントのインデックス作成にCognitive Searchを利用
- エージェントの知能化にAzure OpenAIを活用
- RAGパターンの実装

### 3. MCPツール
- GitHubリポジトリの解析
- コードの検査
- メタデータの抽出

## Code Walkthrough

このサンプルで示す内容：
1. MCPサーバーとの連携
2. マルチエージェントのオーケストレーション
3. Azure Cognitive Searchの統合
4. RAGパターンの実装

主な特徴：
- リアルタイムのGitHubリポジトリ解析
- インテリジェントなプロジェクト推薦
- Azure Searchを用いたイベントマッチング
- Chainlitによるストリーミングレスポンス

## Running the Sample

1. MCPサーバーを起動する：
   ```bash
   npx @modelcontextprotocol/server-github
   ```

2. アプリケーションを起動する：
   ```bash
   chainlit run app.py -w
   ```

3. 統合テストを行う：
   ```
   Example query: "Analyze repositories for username: <github_username>"
   ```

## Troubleshooting

よくある問題と対処法：
1. MCP接続の問題
   - サーバーが起動しているか確認
   - ポートの使用状況をチェック
   - GitHubトークンを確認

2. Azure Searchの問題
   - 接続文字列の検証
   - インデックスの存在確認
   - ドキュメントのアップロード状況確認

## Next Steps
- さらなるMCPツールの活用を探る
- カスタムエージェントを実装する
- RAG機能を強化する
- イベントソースを追加する

## Resources
- [MCP for Beginners](https://aka.ms/mcp-for-beginners)  
- [MCP Documentation](https://github.com/microsoft/semantic-kernel/tree/main/python/semantic-kernel/semantic_kernel/connectors/mcp)
- [Azure Cognitive Search Docs](https://learn.microsoft.com/azure/search/)
- [Semantic Kernel Guides](https://learn.microsoft.com/semantic-kernel/)

**免責事項**：  
本書類はAI翻訳サービス「[Co-op Translator](https://github.com/Azure/co-op-translator)」を使用して翻訳されています。正確性には努めておりますが、自動翻訳には誤りや不正確な箇所が含まれる可能性があることをご了承ください。原文の言語によるオリジナル文書が正式な情報源とみなされます。重要な情報については、専門の人間による翻訳を推奨します。本翻訳の利用により生じたいかなる誤解や誤訳についても責任を負いかねます。