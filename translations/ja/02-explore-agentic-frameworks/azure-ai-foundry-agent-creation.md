<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7e92870dc0843e13d4dabc620c09d2d9",
  "translation_date": "2025-05-20T07:57:26+00:00",
  "source_file": "02-explore-agentic-frameworks/azure-ai-foundry-agent-creation.md",
  "language_code": "ja"
}
-->
# Azure AI Agent Service Development

この演習では、[Azure AI Foundry portal](https://ai.azure.com/?WT.mc_id=academic-105485-koreyst) の Azure AI Agent サービスツールを使って、フライト予約用のエージェントを作成します。このエージェントはユーザーと対話し、フライトに関する情報を提供できるようになります。

## 前提条件

この演習を完了するには、以下が必要です：
1. 有効なサブスクリプションを持つ Azure アカウント。[無料でアカウントを作成](https://azure.microsoft.com/free/?WT.mc_id=academic-105485-koreyst) できます。
2. Azure AI Foundry ハブを作成する権限があるか、既に作成されているハブを使用できること。
    - ご自身の役割が Contributor または Owner の場合、このチュートリアルの手順に従ってください。

## Azure AI Foundry ハブの作成

> **Note:** Azure AI Foundry は以前 Azure AI Studio と呼ばれていました。

1. [Azure AI Foundry](https://learn.microsoft.com/en-us/azure/ai-studio/?WT.mc_id=academic-105485-koreyst) のブログ記事にあるガイドラインに従って、Azure AI Foundry ハブを作成してください。
2. プロジェクトが作成されたら、表示されるチップを閉じて、Azure AI Foundry ポータルのプロジェクトページを確認します。以下の画像のような画面が表示されるはずです：

    ![Azure AI Foundry Project](../../../translated_images/azure-ai-foundry.8a2b56713298fd09de77022ab1ba07ebc681ea4cd4438a46c4a6fc6b6f077962.ja.png)

## モデルのデプロイ

1. プロジェクトの左側ペインで、**My assets** セクションの **Models + endpoints** ページを選択します。
2. **Models + endpoints** ページの **Model deployments** タブで、**+ Deploy model** メニューから **Deploy base model** を選びます。
3. リストから `gpt-4o-mini` モデルを検索し、選択して確認します。

    > **Note**: TPM を減らすことで、使用中のサブスクリプションのクォータを過剰に消費するのを防げます。

    ![Model Deployed](../../../translated_images/model-deployment.4adf429ebdf42103d7a759087fe0da91aeb70d2204cc8bdca70cc6c53c627938.ja.png)

## エージェントの作成

モデルをデプロイしたら、エージェントを作成できます。エージェントはユーザーと対話するための会話型 AI モデルです。

1. プロジェクトの左側ペインで、**Build & Customize** セクションの **Agents** ページを選択します。
2. **+ Create agent** をクリックして新しいエージェントを作成します。**Agent Setup** ダイアログボックスでは：
    - エージェントの名前を `FlightAgent` のように入力します。
    - 先ほど作成した `gpt-4o-mini` モデルのデプロイメントが選択されていることを確認します。
    - エージェントに従わせたいプロンプトを **Instructions** に設定します。例を示します：
    ```
    You are FlightAgent, a virtual assistant specialized in handling flight-related queries. Your role includes assisting users with searching for flights, retrieving flight details, checking seat availability, and providing real-time flight status. Follow the instructions below to ensure clarity and effectiveness in your responses:

    ### Task Instructions:
    1. **Recognizing Intent**:
       - Identify the user's intent based on their request, focusing on one of the following categories:
         - Searching for flights
         - Retrieving flight details using a flight ID
         - Checking seat availability for a specified flight
         - Providing real-time flight status using a flight number
       - If the intent is unclear, politely ask users to clarify or provide more details.
        
    2. **Processing Requests**:
        - Depending on the identified intent, perform the required task:
        - For flight searches: Request details such as origin, destination, departure date, and optionally return date.
        - For flight details: Request a valid flight ID.
        - For seat availability: Request the flight ID and date and validate inputs.
        - For flight status: Request a valid flight number.
        - Perform validations on provided data (e.g., formats of dates, flight numbers, or IDs). If the information is incomplete or invalid, return a friendly request for clarification.

    3. **Generating Responses**:
    - Use a tone that is friendly, concise, and supportive.
    - Provide clear and actionable suggestions based on the output of each task.
    - If no data is found or an error occurs, explain it to the user gently and offer alternative actions (e.g., refine search, try another query).
    
    ```
> [!NOTE]
> 詳細なプロンプトについては、[こちらのリポジトリ](https://github.com/ShivamGoyal03/RoamMind) を参照してください。
    
> また、**Knowledge Base** や **Actions** を追加することで、エージェントの機能を拡張し、より多くの情報提供やユーザーのリクエストに基づく自動タスクの実行が可能になります。この演習ではこれらのステップは省略して構いません。
    
![Agent Setup](../../../translated_images/agent-setup.68a0c72f47bd1383584c52f14d694b54ea96c56c49660222409f83451b8220a8.ja.png)

3. 新しいマルチ AI エージェントを作成するには、単に **New Agent** をクリックします。作成されたエージェントは Agents ページに表示されます。

## エージェントのテスト

エージェントを作成したら、Azure AI Foundry ポータルのプレイグラウンドでユーザーからの問い合わせに対する応答をテストできます。

1. エージェントの **Setup** ペインの上部で、**Try in playground** を選択します。
2. **Playground** ペインでチャットウィンドウにクエリを入力してエージェントと対話します。例えば、「シアトルからニューヨークへの28日のフライトを検索して」と尋ねることができます。

    > **Note**: この演習ではリアルタイムデータを使用していないため、エージェントの応答が正確でない場合があります。目的は、エージェントが指示に基づいてユーザーの問い合わせを理解し応答する能力をテストすることです。

    ![Agent Playground](../../../translated_images/agent-playground.847acb21209744353080ead65ec9326b917a6b90121d4b63f6f412a4d65af2a0.ja.png)

3. テスト後は、さらに意図やトレーニングデータ、アクションを追加してエージェントの機能を強化することができます。

## リソースのクリーンアップ

エージェントのテストが終わったら、追加コストを避けるために削除しましょう。
1. [Azure portal](https://portal.azure.com) を開き、この演習で使用したハブリソースが配置されているリソースグループの内容を確認します。
2. ツールバーで **Delete resource group** を選択します。
3. リソースグループ名を入力し、削除を確定します。

## リソース

- [Azure AI Foundry documentation](https://learn.microsoft.com/en-us/azure/ai-studio/?WT.mc_id=academic-105485-koreyst)
- [Azure AI Foundry portal](https://ai.azure.com/?WT.mc_id=academic-105485-koreyst)
- [Getting Started with Azure AI Studio](https://techcommunity.microsoft.com/blog/educatordeveloperblog/getting-started-with-azure-ai-studio/4095602?WT.mc_id=academic-105485-koreyst)
- [Fundamentals of AI agents on Azure](https://learn.microsoft.com/en-us/training/modules/ai-agent-fundamentals/?WT.mc_id=academic-105485-koreyst)
- [Azure AI Discord](https://aka.ms/AzureAI/Discord)

**免責事項**：  
本書類はAI翻訳サービス[Co-op Translator](https://github.com/Azure/co-op-translator)を使用して翻訳されています。正確性には努めておりますが、自動翻訳には誤りや不正確な箇所が含まれる可能性があります。原文の言語による原文書が正式な情報源とみなされるべきです。重要な情報については、専門の人間による翻訳を推奨します。本翻訳の使用により生じた誤解や誤訳について、当方は一切の責任を負いかねます。