<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "88258b03f2893aa2e69eb8fb24baabbc",
  "translation_date": "2025-05-20T08:23:13+00:00",
  "source_file": "04-tool-use/README.md",
  "language_code": "ja"
}
-->
[![How to Design Good AI Agents](../../../translated_images/lesson-4-thumbnail.2c292cd87b951b3e914e9548b46cb4d14a0852f9c8d75e9566d46da839c983d9.ja.png)](https://youtu.be/vieRiPRx-gI?si=cEZ8ApnT6Sus9rhn)

> _(上の画像をクリックすると、このレッスンの動画が視聴できます)_

# ツール利用デザインパターン

ツールは、AIエージェントにより幅広い能力を持たせることができるため興味深い存在です。エージェントが実行できるアクションの種類が限られている代わりに、ツールを追加することで、エージェントは多様なアクションを実行可能になります。本章では、AIエージェントが特定のツールを使って目標を達成する方法を示す「ツール利用デザインパターン」について解説します。

## はじめに

このレッスンでは、以下の質問に答えていきます：

- ツール利用デザインパターンとは何か？
- どのようなユースケースに適用できるか？
- デザインパターンを実装するために必要な要素・構成要素は何か？
- 信頼できるAIエージェントを構築するためにツール利用デザインパターンを使う際の特別な注意点は何か？

## 学習目標

このレッスンを終えた後、あなたは以下のことができるようになります：

- ツール利用デザインパターンの定義と目的を説明できる
- 適用可能なユースケースを特定できる
- デザインパターンを実装するための主要な要素を理解できる
- このデザインパターンを使う際の信頼性確保のための考慮点を認識できる

## ツール利用デザインパターンとは？

**ツール利用デザインパターン**は、LLMに外部ツールと連携して特定の目標を達成する能力を持たせることに焦点を当てています。ツールとは、エージェントが実行できるコードであり、例えば計算機のような単純な関数や、株価照会や天気予報などのサードパーティサービスへのAPIコールなどがあります。AIエージェントの文脈では、ツールは**モデル生成の関数呼び出し**に応じて実行されるよう設計されています。

## どのようなユースケースに適用できるか？

AIエージェントはツールを活用して複雑なタスクを完遂したり、情報を取得したり、意思決定を行ったりできます。ツール利用デザインパターンは、データベース、ウェブサービス、コードインタプリタなど外部システムと動的にやりとりする必要があるシナリオでよく使われます。具体的には以下のようなユースケースに役立ちます：

- **動的情報取得**：外部APIやデータベースに問い合わせて最新データを取得（例：SQLiteデータベースからのデータ分析、株価や天気情報の取得）
- **コード実行と解釈**：コードやスクリプトを実行して数学的問題を解く、レポートを生成する、シミュレーションを行う
- **ワークフロー自動化**：タスクスケジューラ、メールサービス、データパイプラインなどのツールを組み合わせて繰り返し作業や複数ステップの処理を自動化
- **カスタマーサポート**：CRMシステム、チケッティングプラットフォーム、ナレッジベースと連携してユーザーの問い合わせに対応
- **コンテンツ生成・編集**：文法チェッカー、テキスト要約器、コンテンツ安全性評価ツールなどを活用してコンテンツ作成を支援

## ツール利用デザインパターンを実装するための要素・構成要素は？

これらの構成要素により、AIエージェントは幅広いタスクを実行可能になります。ツール利用デザインパターンを実装するために必要な主要な要素を見ていきましょう：

- **関数／ツールスキーマ**：利用可能なツールの詳細な定義（関数名、目的、必須パラメーター、期待される出力など）。これによりLLMはどのツールが使えるかを理解し、有効なリクエストを構築できるようになります。

- **関数実行ロジック**：ユーザーの意図や会話の文脈に基づいて、いつどのツールを呼び出すかを制御します。プランナーやルーティング機構、条件分岐など、動的にツール利用を判断する仕組みが含まれます。

- **メッセージ処理システム**：ユーザー入力、LLMの応答、ツール呼び出し、ツールの出力間の会話フローを管理するコンポーネント。

- **ツール統合フレームワーク**：単純な関数から複雑な外部サービスまで、エージェントと各種ツールを接続するインフラ。

- **エラーハンドリング＆検証**：ツール実行の失敗処理、パラメーターの検証、予期しない応答の管理。

- **状態管理**：会話の文脈、過去のツール利用履歴、永続データを追跡し、マルチターンのやり取りで一貫性を保つ。

次に、関数／ツール呼び出しについて詳しく見ていきましょう。

### 関数／ツール呼び出し

関数呼び出しは、LLMがツールと連携するための主要な方法です。『関数』と『ツール』はしばしば同義で使われます。なぜなら、『関数』（再利用可能なコードのブロック）がエージェントがタスクを実行するための『ツール』だからです。関数のコードを呼び出すためには、LLMがユーザーのリクエストを関数の説明と照合する必要があります。そのため、利用可能な関数の説明を含むスキーマをLLMに渡します。LLMはタスクに最適な関数を選択し、その名前と引数を返します。選ばれた関数が実行され、その応答がLLMに返され、LLMはその情報を使ってユーザーのリクエストに応答します。

開発者がエージェントの関数呼び出しを実装するために必要なものは：

1. 関数呼び出しをサポートするLLMモデル
2. 関数説明を含むスキーマ
3. 各関数のコード

例えば、ある都市の現在時刻を取得する例を見てみましょう：

1. **関数呼び出しをサポートするLLMを初期化する：**

    すべてのモデルが関数呼び出しをサポートしているわけではないため、使用しているLLMが対応しているか確認が必要です。<a href="https://learn.microsoft.com/azure/ai-services/openai/how-to/function-calling" target="_blank">Azure OpenAI</a>は関数呼び出しをサポートしています。まずAzure OpenAIクライアントを初期化しましょう。

    ```python
    # Initialize the Azure OpenAI client
    client = AzureOpenAI(
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
        api_version="2024-05-01-preview"
    )
    ```

1. **関数スキーマを作成する：**

    次に、関数名、関数の説明、関数パラメーターの名前と説明を含むJSONスキーマを定義します。
    そのスキーマを先ほど作成したクライアントに渡し、ユーザーの「サンフランシスコの時刻を知りたい」というリクエストと一緒に送ります。重要なのは、**ツール呼び出し**が返されることであり、質問の最終的な答えではありません。先述の通り、LLMはタスクに選んだ関数名と引数を返します。

    ```python
    # Function description for the model to read
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get the current time in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name, e.g. San Francisco",
                        },
                    },
                    "required": ["location"],
                },
            }
        }
    ]
    ```
   
    ```python
  
    # Initial user message
    messages = [{"role": "user", "content": "What's the current time in San Francisco"}] 
  
    # First API call: Ask the model to use the function
      response = client.chat.completions.create(
          model=deployment_name,
          messages=messages,
          tools=tools,
          tool_choice="auto",
      )
  
      # Process the model's response
      response_message = response.choices[0].message
      messages.append(response_message)
  
      print("Model's response:")  

      print(response_message)
  
    ```

    ```bash
    Model's response:
    ChatCompletionMessage(content=None, role='assistant', function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_pOsKdUlqvdyttYB67MOj434b', function=Function(arguments='{"location":"San Francisco"}', name='get_current_time'), type='function')])
    ```
  
1. **タスクを実行する関数コード：**

    LLMが実行すべき関数を選んだので、そのタスクを実行するコードを実装して実行します。
    Pythonで現在時刻を取得するコードを実装しましょう。また、response_messageから関数名と引数を抽出し、最終結果を得るコードも必要です。

    ```python
      def get_current_time(location):
        """Get the current time for a given location"""
        print(f"get_current_time called with location: {location}")  
        location_lower = location.lower()
        
        for key, timezone in TIMEZONE_DATA.items():
            if key in location_lower:
                print(f"Timezone found for {key}")  
                current_time = datetime.now(ZoneInfo(timezone)).strftime("%I:%M %p")
                return json.dumps({
                    "location": location,
                    "current_time": current_time
                })
      
        print(f"No timezone data found for {location_lower}")  
        return json.dumps({"location": location, "current_time": "unknown"})
    ```

    ```python
     # Handle function calls
      if response_message.tool_calls:
          for tool_call in response_message.tool_calls:
              if tool_call.function.name == "get_current_time":
     
                  function_args = json.loads(tool_call.function.arguments)
     
                  time_response = get_current_time(
                      location=function_args.get("location")
                  )
     
                  messages.append({
                      "tool_call_id": tool_call.id,
                      "role": "tool",
                      "name": "get_current_time",
                      "content": time_response,
                  })
      else:
          print("No tool calls were made by the model.")  
  
      # Second API call: Get the final response from the model
      final_response = client.chat.completions.create(
          model=deployment_name,
          messages=messages,
      )
  
      return final_response.choices[0].message.content
     ```

    ```bash
      get_current_time called with location: San Francisco
      Timezone found for san francisco
      The current time in San Francisco is 09:24 AM.
     ```

関数呼び出しはほとんどのエージェントのツール利用デザインの中心ですが、一から実装するのは難しいこともあります。
[レッスン2](../../../02-explore-agentic-frameworks)で学んだように、エージェントフレームワークはツール利用を実装するための既製の構成要素を提供してくれます。

## エージェントフレームワークを使ったツール利用の例

ここでは、さまざまなエージェントフレームワークを使ってツール利用デザインパターンを実装する例を紹介します。

### Semantic Kernel

<a href="https://learn.microsoft.com/azure/ai-services/agents/overview" target="_blank">Semantic Kernel</a>は、.NET、Python、Javaの開発者向けのオープンソースAIフレームワークで、LLMを使う際の関数呼び出しを簡単にします。関数とパラメーターをモデルに自動的に説明する「<a href="https://learn.microsoft.com/semantic-kernel/concepts/ai-services/chat-completion/function-calling/?pivots=programming-language-python#1-serializing-the-functions" target="_blank">シリアライズ</a>」というプロセスを通じてこれを実現します。また、モデルとコード間のやりとりも管理します。Semantic Kernelのようなエージェントフレームワークの利点は、<a href="https://github.com/microsoft/semantic-kernel/blob/main/python/samples/getting_started_with_agents/openai_assistant/step4_assistant_tool_file_search.py" target="_blank">File Search</a>や<a href="https://github.com/microsoft/semantic-kernel/blob/main/python/samples/getting_started_with_agents/openai_assistant/step3_assistant_tool_code_interpreter.py" target="_blank">Code Interpreter</a>のような既製ツールにアクセスできることです。

以下の図はSemantic Kernelでの関数呼び出しの流れを示しています：

![function calling](../../../translated_images/functioncalling-diagram.b5493ea5154ad8e3e4940d2e36a49101eec1398948e5d1039942203b4f5a4209.ja.png)

Semantic Kernelでは関数／ツールを<a href="https://learn.microsoft.com/semantic-kernel/concepts/plugins/?pivots=programming-language-python" target="_blank">プラグイン</a>と呼びます。`get_current_time` function we saw earlier into a plugin by turning it into a class with the function in it. We can also import the `kernel_function`デコレーターは関数の説明を受け取り、GetCurrentTimePluginを使ってカーネルを作成すると、関数とパラメーターを自動的にシリアライズし、LLMに送るスキーマを生成します。

```python
from semantic_kernel.functions import kernel_function

class GetCurrentTimePlugin:
    async def __init__(self, location):
        self.location = location

    @kernel_function(
        description="Get the current time for a given location"
    )
    def get_current_time(location: str = ""):
        ...

```

```python 
from semantic_kernel import Kernel

# Create the kernel
kernel = Kernel()

# Create the plugin
get_current_time_plugin = GetCurrentTimePlugin(location)

# Add the plugin to the kernel
kernel.add_plugin(get_current_time_plugin)
```
  
### Azure AI Agent Service

<a href="https://learn.microsoft.com/azure/ai-services/agents/overview" target="_blank">Azure AI Agent Service</a>は、開発者が基盤となる計算・ストレージリソースを管理せずに、高品質で拡張性のあるAIエージェントを安全に構築・展開・スケールできるよう設計された新しいエージェントフレームワークです。エンタープライズ用途に特に適しており、エンタープライズグレードのセキュリティを備えたフルマネージドサービスです。

LLM APIを直接使う場合と比較して、Azure AI Agent Serviceには以下の利点があります：

- 自動ツール呼び出し：ツール呼び出しの解析、実行、応答処理がサーバー側で行われるため不要
- 安全に管理されたデータ：会話状態を自分で管理する代わりに、スレッドで必要な情報をすべて保存可能
- すぐに使えるツール：Bing、Azure AI Search、Azure Functionsなどのデータソースと連携するツールを利用可能

Azure AI Agent Serviceで利用可能なツールは大きく2つに分類されます：

1. ナレッジツール：
    - <a href="https://learn.microsoft.com/azure/ai-services/agents/how-to/tools/bing-grounding?tabs=python&pivots=overview" target="_blank">Bing検索による基盤付け</a>
    - <a href="https://learn.microsoft.com/azure/ai-services/agents/how-to/tools/file-search?tabs=python&pivots=overview" target="_blank">ファイル検索</a>
    - <a href="https://learn.microsoft.com/azure/ai-services/agents/how-to/tools/azure-ai-search?tabs=azurecli%2Cpython&pivots=overview-azure-ai-search" target="_blank">Azure AI Search</a>

2. アクションツール：
    - <a href="https://learn.microsoft.com/azure/ai-services/agents/how-to/tools/function-calling?tabs=python&pivots=overview" target="_blank">関数呼び出し</a>
    - <a href="https://learn.microsoft.com/azure/ai-services/agents/how-to/tools/code-interpreter?tabs=python&pivots=overview" target="_blank">コードインタプリタ</a>
    - <a href="https://learn.microsoft.com/azure/ai-services/agents/how-to/tools/openapi-spec?tabs=python&pivots=overview" target="_blank">OpenAI定義ツール</a>
    - <a href="https://learn.microsoft.com/azure/ai-services/agents/how-to/tools/azure-functions?pivots=overview" target="_blank">Azure Functions</a>

Agent Serviceでは、これらのツールを`toolset`. It also utilizes `threads` which keep track of the history of messages from a particular conversation.

Imagine you are a sales agent at a company called Contoso. You want to develop a conversational agent that can answer questions about your sales data.

The following image illustrates how you could use Azure AI Agent Service to analyze your sales data:

![Agentic Service In Action](../../../translated_images/agent-service-in-action.8c2d8aa8e9d91feeb29549b3fde529f8332b243875154d03907616a69198afbc.ja.jpg)

To use any of these tools with the service we can create a client and define a tool or toolset. To implement this practically we can use the following Python code. The LLM will be able to look at the toolset and decide whether to use the user created function, `fetch_sales_data_using_sqlite_query`のように組み合わせたり、ユーザーリクエストに応じて既製のCode Interpreterを使ったりできます。

```python 
import os
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from fecth_sales_data_functions import fetch_sales_data_using_sqlite_query # fetch_sales_data_using_sqlite_query function which can be found in a fetch_sales_data_functions.py file.
from azure.ai.projects.models import ToolSet, FunctionTool, CodeInterpreterTool

project_client = AIProjectClient.from_connection_string(
    credential=DefaultAzureCredential(),
    conn_str=os.environ["PROJECT_CONNECTION_STRING"],
)

# Initialize function calling agent with the fetch_sales_data_using_sqlite_query function and adding it to the toolset
fetch_data_function = FunctionTool(fetch_sales_data_using_sqlite_query)
toolset = ToolSet()
toolset.add(fetch_data_function)

# Initialize Code Interpreter tool and adding it to the toolset. 
code_interpreter = code_interpreter = CodeInterpreterTool()
toolset = ToolSet()
toolset.add(code_interpreter)

agent = project_client.agents.create_agent(
    model="gpt-4o-mini", name="my-agent", instructions="You are helpful agent", 
    toolset=toolset
)
```

## 信頼できるAIエージェントを構築するためにツール利用デザインパターンを使う際の特別な注意点は？

LLMによって動的に生成されるSQLに関する一般的な懸念はセキュリティです。特にSQLインジェクションや、データベースの削除・改ざんなどの悪意ある操作のリスクです。これらの懸念は、適切なデータベースアクセス権限の設定によって効果的に軽減できます。多くのデータベースでは読み取り専用に設定することが一般的です。PostgreSQLやAzure SQLのようなデータベースサービスでは、アプリには読み取り専用（SELECT）ロールを割り当てます。

さらに、アプリを安全な環境で実行することも保護を強化します。エンタープライズ環境では、運用システムから抽出・変換したデータを読み取り専用のデータベースやデータウェアハウスに格納し、ユーザーフレンドリーなスキーマを設計します。この方法により、データの安全性、パフォーマンスやアクセス性の最適化、アプリの制限付き読み取り専用アクセスが確保されます。

## 追加リソース

- <a href="https://microsoft.github.io/build-your-first-agent-with-azure-ai-agent-service-workshop/" target="_blank">Azure AI Agents Service ワークショップ</a>
- <a href="https://github.com/Azure-Samples/contoso-creative-writer/tree/main/docs/workshop" target="_blank">Contoso Creative Writer マルチエージェント ワークショップ</a>
- <a href="https://learn.microsoft.com/semantic-kernel/concepts/ai-services/chat-completion/function-calling/?pivots=programming-language-python#1-serializing-the-functions" target="_blank">Semantic Kernel 関数呼び出しチュートリアル</a>
- <a href="https://github.com/microsoft/semantic-kernel/blob/main/python/samples/getting_started_with_agents/openai_assistant/step3_assistant_tool_code_interpreter.py" target="_blank">Semantic Kernel コードインタプリタ</a>
- <a href="https://microsoft.github.io/autogen/dev/user-guide/core-user-guide/components/tools.html" target="_blank">Autogen ツール</a>

## 前のレッスン

[Agenticデザインパターンの理解](../03-agentic-design-patterns/README.md)

## 次のレッスン

[Agentic RAG](../05-agentic-rag/README.md)

**免責事項**：  
本書類はAI翻訳サービス[Co-op Translator](https://github.com/Azure/co-op-translator)を使用して翻訳されています。正確性の確保に努めておりますが、自動翻訳には誤りや不正確な部分が含まれる可能性があります。原文の言語で記載された文書が正式な情報源とみなされるべきです。重要な情報については、専門の人間による翻訳を推奨します。本翻訳の利用により生じた誤解や解釈の相違について、当方は一切の責任を負いかねます。