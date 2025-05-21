<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9320dd53c82869fd44935d1581eaf7bb",
  "translation_date": "2025-05-21T08:10:10+00:00",
  "source_file": "11-mcp/README.md",
  "language_code": "zh"
}
-->
# Lesson 11: Model Context Protocol (MCP) 集成

## 学习目标
- 了解 MCP 是什么以及它在 AI 代理开发中的作用
- 设置并配置用于 GitHub 集成的 MCP 服务器
- 使用 MCP 工具构建多代理系统
- 使用 Azure 认知搜索实现 RAG（检索增强生成）

## 前置条件
- Python 3.8 及以上版本
- Node.js 14 及以上版本
- Azure 订阅
- GitHub 账号
- 对 Semantic Kernel 有基本了解

## 设置说明

1. **环境搭建**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **配置 Azure 服务**
   - 创建 Azure 认知搜索资源
   - 设置 Azure OpenAI 服务
   - 在 `.env` 文件中配置环境变量

3. **MCP 服务器搭建**
   ```bash
   npm install -g @modelcontextprotocol/server-github
   ```

## 项目结构

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

## 核心组件

### 1. 多代理系统
- GitHub Agent：仓库分析
- Hackathon Agent：项目推荐
- Events Agent：技术活动推荐

### 2. Azure 集成
- 使用认知搜索进行活动索引
- 通过 Azure OpenAI 提升代理智能
- 实现 RAG 模式

### 3. MCP 工具
- GitHub 仓库分析
- 代码检查
- 元数据提取

## 代码讲解

示例演示：
1. MCP 服务器集成
2. 多代理协调
3. Azure 认知搜索集成
4. RAG 模式实现

主要功能：
- 实时 GitHub 仓库分析
- 智能项目推荐
- 利用 Azure 搜索进行活动匹配
- 使用 Chainlit 实现流式响应

## 运行示例

1. 启动 MCP 服务器：
   ```bash
   npx @modelcontextprotocol/server-github
   ```

2. 启动应用程序：
   ```bash
   chainlit run app.py -w
   ```

3. 测试集成效果：
   ```
   Example query: "Analyze repositories for username: <github_username>"
   ```

## 故障排除

常见问题及解决方案：
1. MCP 连接问题
   - 确认服务器已启动
   - 检查端口是否可用
   - 确认 GitHub 令牌有效

2. Azure 搜索问题
   - 验证连接字符串
   - 检查索引是否存在
   - 确认文档上传成功

## 后续步骤
- 探索更多 MCP 工具
- 实现自定义代理
- 增强 RAG 功能
- 添加更多活动来源

## 资源
- [MCP for Beginners](https://aka.ms/mcp-for-beginners)  
- [MCP Documentation](https://github.com/microsoft/semantic-kernel/tree/main/python/semantic-kernel/semantic_kernel/connectors/mcp)
- [Azure Cognitive Search Docs](https://learn.microsoft.com/azure/search/)
- [Semantic Kernel Guides](https://learn.microsoft.com/semantic-kernel/)

**免责声明**：  
本文件使用 AI 翻译服务 [Co-op Translator](https://github.com/Azure/co-op-translator) 进行翻译。尽管我们力求准确，但请注意，自动翻译可能包含错误或不准确之处。原始文件的母语版本应被视为权威来源。对于重要信息，建议采用专业人工翻译。对于因使用本翻译而产生的任何误解或误释，我们概不负责。