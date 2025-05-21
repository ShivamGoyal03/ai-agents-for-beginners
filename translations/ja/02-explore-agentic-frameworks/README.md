<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "da3523bf8fa456371e21d8d14c67305d",
  "translation_date": "2025-05-20T07:58:56+00:00",
  "source_file": "02-explore-agentic-frameworks/README.md",
  "language_code": "ja"
}
-->
Wikipediaによると、アクターとは「並行計算の基本的な構成要素であり、受け取ったメッセージに応じて、ローカルの意思決定を行い、より多くのアクターを作成し、さらにメッセージを送信し、次に受け取るメッセージにどう応答するかを決定することができる」とされています。

**ユースケース**: コード生成の自動化、データ分析タスク、計画やリサーチ機能のためのカスタムエージェント構築。

AutoGenの重要なコアコンセプトは以下の通りです：

- **エージェント**。エージェントは以下の特徴を持つソフトウェアエンティティです：
  - **メッセージを介して通信**し、これらのメッセージは同期的または非同期的であることができます。
  - **自身の状態を保持**し、受信したメッセージによって状態が変更されることがあります。
  - **受信したメッセージや状態の変化に応じてアクションを実行**します。これらのアクションはエージェントの状態を変更したり、メッセージログの更新、新たなメッセージの送信、コードの実行、API呼び出しなどの外部効果をもたらすことがあります。

  以下はチャット機能を持つ独自エージェントを作成する簡単なコードスニペットです：

    ```python
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.messages import TextMessage
    from autogen_ext.models.openai import OpenAIChatCompletionClient


    class MyAssistant(RoutedAgent):
        def __init__(self, name: str) -> None:
            super().__init__(name)
            model_client = OpenAIChatCompletionClient(model="gpt-4o")
            self._delegate = AssistantAgent(name, model_client=model_client)
    
        @message_handler
        async def handle_my_message_type(self, message: MyMessageType, ctx: MessageContext) -> None:
            print(f"{self.id.type} received message: {message.content}")
            response = await self._delegate.on_messages(
                [TextMessage(content=message.content, source="user")], ctx.cancellation_token
            )
            print(f"{self.id.type} responded: {response.chat_message.content}")
    ```
    
    上記のコードでは、`MyAssistant` has been created and inherits from `RoutedAgent`. It has a message handler that prints the content of the message and then sends a response using the `AssistantAgent` delegate. Especially note how we assign to `self._delegate` an instance of `AssistantAgent`はチャット完了処理を扱える事前構築済みエージェントです。

    次に、このエージェントタイプをAutoGenに登録し、プログラムを開始します：

    ```python
    
    # main.py
    runtime = SingleThreadedAgentRuntime()
    await MyAgent.register(runtime, "my_agent", lambda: MyAgent())

    runtime.start()  # Start processing messages in the background.
    await runtime.send_message(MyMessageType("Hello, World!"), AgentId("my_agent", "default"))
    ```

    上記のコードではエージェントがランタイムに登録され、メッセージがエージェントに送信されることで以下の出力が得られます：

    ```text
    # Output from the console:
    my_agent received message: Hello, World!
    my_assistant received message: Hello, World!
    my_assistant responded: Hello! How can I assist you today?
    ```

- **マルチエージェント**。AutoGenは複数のエージェントが協力して複雑なタスクを達成できるようサポートしています。エージェントは通信し、情報を共有し、行動を調整して問題解決を効率化します。マルチエージェントシステムを作るには、データ取得、分析、意思決定、ユーザーインタラクションなど専門的な役割を持つ異なるタイプのエージェントを定義します。以下はその一例です：

    ```python
    editor_description = "Editor for planning and reviewing the content."

    # Example of declaring an Agent
    editor_agent_type = await EditorAgent.register(
    runtime,
    editor_topic_type,  # Using topic type as the agent type.
    lambda: EditorAgent(
        description=editor_description,
        group_chat_topic_type=group_chat_topic_type,
        model_client=OpenAIChatCompletionClient(
            model="gpt-4o-2024-08-06",
            # api_key="YOUR_API_KEY",
        ),
        ),
    )

    # remaining declarations shortened for brevity

    # Group chat
    group_chat_manager_type = await GroupChatManager.register(
    runtime,
    "group_chat_manager",
    lambda: GroupChatManager(
        participant_topic_types=[writer_topic_type, illustrator_topic_type, editor_topic_type, user_topic_type],
        model_client=OpenAIChatCompletionClient(
            model="gpt-4o-2024-08-06",
            # api_key="YOUR_API_KEY",
        ),
        participant_descriptions=[
            writer_description, 
            illustrator_description, 
            editor_description, 
            user_description
        ],
        ),
    )
    ```

    上記のコードでは、`GroupChatManager`がランタイムに登録されており、このマネージャーはライター、イラストレーター、編集者、ユーザーなど異なるタイプのエージェント間のインタラクションを調整しています。

- **エージェントランタイム**。このフレームワークはエージェント間の通信を可能にし、エージェントの識別やライフサイクルを管理し、セキュリティやプライバシーの境界を強制するランタイム環境を提供します。つまり、安全かつ制御された環境でエージェントを実行し、効率的かつ安全に相互作用できるようにします。主に以下の2種類のランタイムがあります：
  - **スタンドアロンランタイム**。すべてのエージェントが同一のプログラミング言語で同じプロセス内に実装される単一プロセスアプリケーションに適しています。以下はその動作イメージです：

アプリケーションスタック

    *エージェントはメッセージを通じてランタイムと通信し、ランタイムはエージェントのライフサイクルを管理します*

  - **分散エージェントランタイム**。エージェントが異なるプログラミング言語で実装され、異なるマシンで動作するマルチプロセスアプリケーションに適しています。以下はその動作イメージです：

## Semantic Kernel + Agent Framework

Semantic Kernelはエンタープライズ向けのAIオーケストレーションSDKです。AIコネクターやメモリコネクターに加え、エージェントフレームワークを備えています。

まず、いくつかのコアコンポーネントを紹介します：

- **AIコネクター**：PythonとC#の両方で利用可能な外部AIサービスやデータソースとのインターフェースです。

  ```python
  # Semantic Kernel Python
  from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
  from semantic_kernel.kernel import Kernel

  kernel = Kernel()
  kernel.add_service(
    AzureChatCompletion(
        deployment_name="your-deployment-name",
        api_key="your-api-key",
        endpoint="your-endpoint",
    )
  )
  ```  

    ```csharp
    // Semantic Kernel C#
    using Microsoft.SemanticKernel;

    // Create kernel
    var builder = Kernel.CreateBuilder();
    
    // Add a chat completion service:
    builder.Services.AddAzureOpenAIChatCompletion(
        "your-resource-name",
        "your-endpoint",
        "your-resource-key",
        "deployment-model");
    var kernel = builder.Build();
    ```

    ここでは、カーネルを作成し、チャット完了サービスを追加する簡単な例を示しています。Semantic KernelはAzure OpenAIチャット完了サービスへの接続を作成しています。

- **プラグイン**：アプリケーションが利用可能な関数をカプセル化します。既製のプラグインもあれば、カスタムプラグインも作成可能です。関連する概念に「プロンプト関数」があります。自然言語での関数呼び出しの代わりに、特定の関数をモデルに公開し、現在のチャットコンテキストに基づいてモデルがこれらの関数のいずれかを呼び出してリクエストやクエリを完了します。例を示します：

  ```python
  from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import AzureChatCompletion


  async def main():
      from semantic_kernel.functions import KernelFunctionFromPrompt
      from semantic_kernel.kernel import Kernel

      kernel = Kernel()
      kernel.add_service(AzureChatCompletion())

      user_input = input("User Input:> ")

      kernel_function = KernelFunctionFromPrompt(
          function_name="SummarizeText",
          prompt="""
          Summarize the provided unstructured text in a sentence that is easy to understand.
          Text to summarize: {{$user_input}}
          """,
      )

      response = await kernel_function.invoke(kernel=kernel, user_input=user_input)
      print(f"Model Response: {response}")

      """
      Sample Console Output:

      User Input:> I like dogs
      Model Response: The text expresses a preference for dogs.
      """


  if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
  ```

    ```csharp
    var userInput = Console.ReadLine();

    // Define semantic function inline.
    string skPrompt = @"Summarize the provided unstructured text in a sentence that is easy to understand.
                        Text to summarize: {{$userInput}}";
    
    // create the function from the prompt
    KernelFunction summarizeFunc = kernel.CreateFunctionFromPrompt(
        promptTemplate: skPrompt,
        functionName: "SummarizeText"
    );

    //then import into the current kernel
    kernel.ImportPluginFromFunctions("SemanticFunctions", [summarizeFunc]);

    ```

    ここでは、`skPrompt` that leaves room for the user to input text, `$userInput`. Then you create the kernel function `SummarizeText` and then import it into the kernel with the plugin name `SemanticFunctions`というテンプレートプロンプトがあります。Semantic Kernelが関数の内容と呼び出しタイミングを理解するのに役立つ関数名に注目してください。

- **ネイティブ関数**：フレームワークが直接呼び出してタスクを実行する関数もあります。以下はファイルから内容を取得するネイティブ関数の例です：

    ```csharp
    public class NativeFunctions {

        [SKFunction, Description("Retrieve content from local file")]
        public async Task<string> RetrieveLocalFile(string fileName, int maxSize = 5000)
        {
            string content = await File.ReadAllTextAsync(fileName);
            if (content.Length <= maxSize) return content;
            return content.Substring(0, maxSize);
        }
    }
    
    //Import native function
    string plugInName = "NativeFunction";
    string functionName = "RetrieveLocalFile";

   //To add the functions to a kernel use the following function
    kernel.ImportPluginFromType<NativeFunctions>();

    ```

- **メモリ**：AIアプリのコンテキスト管理を抽象化し簡素化します。メモリとは、LLMが知っておくべき情報のことです。この情報はベクターストア（インメモリデータベースやベクターデータベースなど）に保存できます。以下は、単純化したシナリオで「事実」をメモリに追加する例です：

    ```csharp
    var facts = new Dictionary<string,string>();
    facts.Add(
        "Azure Machine Learning; https://learn.microsoft.com/azure/machine-learning/",
        @"Azure Machine Learning is a cloud service for accelerating and
        managing the machine learning project lifecycle. Machine learning professionals,
        data scientists, and engineers can use it in their day-to-day workflows"
    );
    
    facts.Add(
        "Azure SQL Service; https://learn.microsoft.com/azure/azure-sql/",
        @"Azure SQL is a family of managed, secure, and intelligent products
        that use the SQL Server database engine in the Azure cloud."
    );
    
    string memoryCollectionName = "SummarizedAzureDocs";
    
    foreach (var fact in facts) {
        await memoryBuilder.SaveReferenceAsync(
            collection: memoryCollectionName,
            description: fact.Key.Split(";")[1].Trim(),
            text: fact.Value,
            externalId: fact.Key.Split(";")[2].Trim(),
            externalSourceName: "Azure Documentation"
        );
    }
    ```

    これらの事実は`SummarizedAzureDocs`というメモリコレクションに保存されます。非常に簡単な例ですが、LLMが利用できる情報をメモリに保存する方法がわかります。

以上がSemantic Kernelフレームワークの基本です。ではエージェントフレームワークはどうでしょうか？

## Azure AI Agent Service

Azure AI Agent ServiceはMicrosoft Ignite 2024で発表された比較的新しいサービスです。Llama 3、Mistral、CohereなどのオープンソースLLMを直接呼び出すなど、より柔軟なモデルでAIエージェントの開発・展開を可能にします。

Azure AI Agent Serviceは強力なエンタープライズ向けセキュリティ機構とデータ保存方法を提供し、企業向けアプリケーションに適しています。

AutoGenやSemantic Kernelのようなマルチエージェントオーケストレーションフレームワークとすぐに連携できます。

現在パブリックプレビュー中で、PythonとC#でのエージェント構築をサポートしています。

Semantic Kernel Pythonを使って、ユーザー定義プラグイン付きのAzure AI Agentを作成できます：

```python
import asyncio
from typing import Annotated

from azure.identity.aio import DefaultAzureCredential

from semantic_kernel.agents import AzureAIAgent, AzureAIAgentSettings, AzureAIAgentThread
from semantic_kernel.contents import ChatMessageContent
from semantic_kernel.contents import AuthorRole
from semantic_kernel.functions import kernel_function


# Define a sample plugin for the sample
class MenuPlugin:
    """A sample Menu Plugin used for the concept sample."""

    @kernel_function(description="Provides a list of specials from the menu.")
    def get_specials(self) -> Annotated[str, "Returns the specials from the menu."]:
        return """
        Special Soup: Clam Chowder
        Special Salad: Cobb Salad
        Special Drink: Chai Tea
        """

    @kernel_function(description="Provides the price of the requested menu item.")
    def get_item_price(
        self, menu_item: Annotated[str, "The name of the menu item."]
    ) -> Annotated[str, "Returns the price of the menu item."]:
        return "$9.99"


async def main() -> None:
    ai_agent_settings = AzureAIAgentSettings.create()

    async with (
        DefaultAzureCredential() as creds,
        AzureAIAgent.create_client(
            credential=creds,
            conn_str=ai_agent_settings.project_connection_string.get_secret_value(),
        ) as client,
    ):
        # Create agent definition
        agent_definition = await client.agents.create_agent(
            model=ai_agent_settings.model_deployment_name,
            name="Host",
            instructions="Answer questions about the menu.",
        )

        # Create the AzureAI Agent using the defined client and agent definition
        agent = AzureAIAgent(
            client=client,
            definition=agent_definition,
            plugins=[MenuPlugin()],
        )

        # Create a thread to hold the conversation
        # If no thread is provided, a new thread will be
        # created and returned with the initial response
        thread: AzureAIAgentThread | None = None

        user_inputs = [
            "Hello",
            "What is the special soup?",
            "How much does that cost?",
            "Thank you",
        ]

        try:
            for user_input in user_inputs:
                print(f"# User: '{user_input}'")
                # Invoke the agent for the specified thread
                response = await agent.get_response(
                    messages=user_input,
                    thread_id=thread,
                )
                print(f"# {response.name}: {response.content}")
                thread = response.thread
        finally:
            await thread.delete() if thread else None
            await client.agents.delete_agent(agent.id)


if __name__ == "__main__":
    asyncio.run(main())
```

### コアコンセプト

Azure AI Agent Serviceには以下のコアコンセプトがあります：

- **エージェント**。Azure AI Agent ServiceはAzure AI Foundryと統合されています。AI Foundry内で、AIエージェントは質問応答（RAG）、アクション実行、ワークフローの完全自動化に使われる「スマート」マイクロサービスとして機能します。生成AIモデルの力と、実世界のデータソースにアクセス・操作するツールを組み合わせて実現します。以下はエージェントの例です：

    ```python
    agent = project_client.agents.create_agent(
        model="gpt-4o-mini",
        name="my-agent",
        instructions="You are helpful agent",
        tools=code_interpreter.definitions,
        tool_resources=code_interpreter.resources,
    )
    ```

    この例では、`gpt-4o-mini`, a name `my-agent`, and instructions `You are helpful agent`というモデルでエージェントが作成されています。コード解釈タスクを実行するためのツールやリソースが装備されています。

- **スレッドとメッセージ**。スレッドはエージェントとユーザー間の会話やインタラクションを表す重要な概念です。スレッドは会話の進行状況を追跡し、コンテキスト情報を保存し、インタラクションの状態を管理します。以下はスレッドの例です：

    ```python
    thread = project_client.agents.create_thread()
    message = project_client.agents.create_message(
        thread_id=thread.id,
        role="user",
        content="Could you please create a bar chart for the operating profit using the following data and provide the file to me? Company A: $1.2 million, Company B: $2.5 million, Company C: $3.0 million, Company D: $1.8 million",
    )
    
    # Ask the agent to perform work on the thread
    run = project_client.agents.create_and_process_run(thread_id=thread.id, agent_id=agent.id)
    
    # Fetch and log all messages to see the agent's response
    messages = project_client.agents.list_messages(thread_id=thread.id)
    print(f"Messages: {messages}")
    ```

    上記コードではスレッドが作成され、その後メッセージが送信されます。`create_and_process_run`を呼ぶことでエージェントがスレッド上で作業を行います。最後にメッセージを取得してログに出力し、エージェントの応答を確認します。メッセージはテキスト、画像、ファイルなど様々なタイプがあり、例えばエージェントの作業結果として画像やテキスト応答が生成されることもあります。開発者はこれらの情報を使って応答をさらに処理したり、ユーザーに提示したりできます。

- **他のAIフレームワークとの統合**。Azure AI Agent ServiceはAutoGenやSemantic Kernelなど他のフレームワークと連携可能です。つまり、アプリの一部をこれらのフレームワークで構築し、Agent Serviceをオーケストレーターとして使ったり、すべてをAgent Serviceで構築したりできます。

**ユースケース**: Azure AI Agent Serviceは、安全でスケーラブルかつ柔軟なAIエージェント展開を必要とするエンタープライズ向けアプリケーションに設計されています。

## これらのフレームワークの違いは？

これらのフレームワークには多くの共通点がありますが、設計、機能、対象ユースケースにおいていくつかの重要な違いがあります：

- **AutoGen**：マルチエージェントシステムに関する先端研究に焦点を当てた実験用フレームワークです。高度なマルチエージェントシステムのプロトタイプ作成や実験に最適です。
- **Semantic Kernel**：エンタープライズ向けの本番運用対応エージェントライブラリです。イベント駆動型で分散したエージェントアプリケーションを構築でき、複数のLLMやSLM、ツール、単一・マルチエージェント設計パターンに対応しています。
- **Azure AI Agent Service**：Azure Foundry上のエージェント用プラットフォーム兼デプロイメントサービスです。Azure OpenAI、Azure AI Search、Bing Search、コード実行などAzureの各種サービスとの接続を提供します。

まだどれを選ぶか迷っていますか？

### ユースケース

よくあるユースケースを通じて選択の参考にしましょう：

> Q: 実験や学習、概念実証のエージェントアプリを素早く構築・試行したい場合は？
>

> A: AutoGenが適しています。イベント駆動型で分散したエージェントアプリケーションに特化し、高度なマルチエージェント設計パターンをサポートしています。

> Q: なぜこのユースケースでAutoGenがSemantic KernelやAzure AI Agent Serviceより良い選択なのですか？
>
> A: AutoGenはイベント駆動型の分散エージェントアプリケーション向けに設計されており、コード生成やデータ分析タスクの自動化に最適なツールと機能を提供します。

> Q: Azure AI Agent Serviceもコード生成などのツールを持っているので、こちらでも良さそうですね？
>
> A: はい、Azure AI Agent Serviceは複数モデルやAzure AI Search、Bing Search、Azure Functionsなどを組み込んだプラットフォームサービスです。Foundryポータルでエージェントを簡単に構築・大規模展開できます。

> Q: まだ迷っています。とりあえず一つだけ教えてください。
>
> A: まずSemantic Kernelでアプリケーションを構築し、その後Azure AI Agent Serviceでデプロイするのが良い選択です。これによりエージェントを簡単に永続化しつつ、Semantic Kernelのマルチエージェント構築機能を活用できます。さらにSemantic KernelはAutoGenにもコネクターがあり、両フレームワークを組み合わせて使うことも容易です。

違いを表にまとめましょう：

| フレームワーク | フォーカス | コアコンセプト | ユースケース |
| --- | --- | --- | --- |
| AutoGen | イベント駆動型、分散エージェントアプリケーション | エージェント、ペルソナ、関数、データ | コード生成、データ分析タスク |
| Semantic Kernel | 人間のようなテキスト理解と生成 | エージェント、モジュラーコンポーネント、コラボレーション | 自然言語理解、コンテンツ生成 |
| Azure AI Agent Service | 柔軟なモデル、エンタープライズセキュリティ、コード生成、ツール呼び出し |  |
## 前のレッスン

[AIエージェントとエージェントのユースケース入門](../01-intro-to-ai-agents/README.md)

## 次のレッスン

[エージェント設計パターンの理解](../03-agentic-design-patterns/README.md)

**免責事項**：  
本書類はAI翻訳サービス[Co-op Translator](https://github.com/Azure/co-op-translator)を使用して翻訳されています。正確性を期しておりますが、自動翻訳には誤りや不正確な部分が含まれる可能性があることをご承知ください。原文の言語によるオリジナル文書が正式な情報源とみなされます。重要な情報については、専門の人間による翻訳を推奨します。本翻訳の使用により生じたいかなる誤解や誤訳についても、当方は責任を負いかねます。