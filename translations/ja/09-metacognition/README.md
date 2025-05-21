<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "8cbf460468c802c7994aa62e0e0779c9",
  "translation_date": "2025-05-20T08:29:02+00:00",
  "source_file": "09-metacognition/README.md",
  "language_code": "ja"
}
-->
[![Multi-Agent Design](../../../translated_images/lesson-9-thumbnail.8ce3844c60ee3125a381e225d70b4f7cde92ae1cc2b2ca5b83137e68e7c20885.ja.png)](https://youtu.be/His9R6gw6Ec?si=3_RMb8VprNvdLRhX)

> _(上の画像をクリックするとこのレッスンの動画が視聴できます)_

# AIエージェントにおけるメタ認知

## はじめに

AIエージェントのメタ認知に関するレッスンへようこそ！この章は、AIエージェントが自分自身の思考過程について考える仕組みに興味がある初心者向けに作られています。このレッスンを終える頃には、重要な概念を理解し、メタ認知をAIエージェント設計に応用するための実践的な例も身につけられます。

## 学習目標

このレッスンを修了すると、以下ができるようになります：

1. エージェント定義における推論ループの意味を理解する。
2. 自己修正可能なエージェントを支援するための計画と評価の技術を使う。
3. タスクを達成するためにコードを操作できるエージェントを自分で作成する。

## メタ認知の紹介

メタ認知とは、自分自身の思考について考える高次の認知プロセスを指します。AIエージェントにとっては、自己認識や過去の経験に基づいて行動を評価・調整できる能力を意味します。メタ認知、つまり「考えることについて考えること」は、エージェント型AIシステムの開発において重要な概念です。AIシステムが自分の内部プロセスを認識し、それを監視、制御、適応できることを含みます。これは、人間が場の空気を読んだり問題を見つめたりするのと似ています。

この自己認識により、AIシステムはより良い判断を下し、誤りを特定し、時間をかけてパフォーマンスを向上させることができます。これらはチューリングテストやAIが支配するかどうかの議論にも繋がります。

エージェント型AIシステムの文脈では、メタ認知は以下のような課題に対処する助けとなります：

- 透明性：AIシステムが自身の推論や判断を説明できることを保証する。
- 推論力：情報を統合し、適切な判断を下す能力を高める。
- 適応性：新しい環境や変化する状況に対応できるようにする。
- 知覚力：環境からのデータを正確に認識・解釈する能力を向上させる。

### メタ認知とは何か？

メタ認知、すなわち「考えることについて考えること」は、自分の認知過程を自己認識し、自己制御する高次の認知プロセスです。AIの領域では、メタ認知はエージェントが自らの戦略や行動を評価・適応する力を与え、問題解決や意思決定能力の向上につながります。

メタ認知を理解することで、より知的で適応性が高く効率的なAIエージェントを設計できます。

真のメタ認知では、AIが自分の推論について明示的に考えます。

例：「安いフライトを優先したけど…直行便を見逃しているかもしれない。もう一度確認しよう。」

どのように、なぜそのルートを選んだかを追跡します。

- 前回のユーザーの好みに過度に依存してミスをしたことを認識し、最終的な提案だけでなく意思決定の戦略自体を修正する。
- 「ユーザーが『混雑しすぎ』と言ったときは、特定の観光地を外すだけでなく、『人気順でトップ観光地を選ぶ方法』に問題があることも反省すべきだ」といったパターンを診断する。

### AIエージェントにおけるメタ認知の重要性

メタ認知はAIエージェント設計において以下の理由で重要な役割を果たします：

![Importance of Metacognition](../../../translated_images/importance-of-metacognition.e351a5983bb745d60a1a60185391a39a6751d033c8c1948ceb6ad04eff7dbeac.ja.png)

- 自己反省：エージェントが自身のパフォーマンスを評価し、改善点を見つけられる。
- 適応性：過去の経験や変化する環境に応じて戦略を変えられる。
- エラー修正：エージェントが自律的に誤りを検出・修正し、より正確な結果を出す。
- リソース管理：時間や計算リソースの使用を最適化するために、行動を計画・評価できる。

## AIエージェントの構成要素

メタ認知プロセスに入る前に、AIエージェントの基本的な構成要素を理解することが重要です。一般的にAIエージェントは以下で構成されます：

- ペルソナ：エージェントの性格や特徴で、ユーザーとのやりとりの仕方を定義する。
- ツール：エージェントが実行できる能力や機能。
- スキル：エージェントが持つ知識や専門性。

これらの要素が組み合わさり、特定のタスクを遂行する「専門ユニット」を形成します。

**例**：旅行代理店のエージェントを考えてみましょう。このエージェントサービスは、休暇の計画だけでなく、リアルタイムのデータや過去の顧客の旅の経験に基づいてプランを調整します。

### 例：旅行代理店サービスにおけるメタ認知

AIによって動く旅行代理店サービスを設計していると想像してください。このエージェント「Travel Agent」はユーザーの休暇計画を支援します。メタ認知を取り入れるために、Travel Agentは自己認識と過去の経験に基づいて行動を評価・調整する必要があります。メタ認知が果たす役割は以下の通りです。

#### 現在のタスク

ユーザーのパリ旅行の計画を手伝うこと。

#### タスク完了の手順

1. **ユーザーの好みを収集**：旅行日程、予算、興味（例：博物館、料理、ショッピング）、特別な要望を聞く。
2. **情報収集**：ユーザーの好みに合うフライト、宿泊施設、観光地、レストランを検索する。
3. **提案作成**：フライト情報、ホテル予約、推奨アクティビティを含むパーソナライズされた旅程を提供する。
4. **フィードバックに基づく調整**：提案に対するユーザーの意見を聞き、必要に応じて調整する。

#### 必要なリソース

- フライトやホテル予約データベースへのアクセス。
- パリの観光地やレストランの情報。
- 過去のユーザーとのやりとりによるフィードバックデータ。

#### 経験と自己反省

Travel Agentはメタ認知を使って自身のパフォーマンスを評価し、過去の経験から学びます。例えば：

1. **ユーザーフィードバックの分析**：どの提案が好評だったかを振り返り、将来の提案に反映させる。
2. **適応性**：過去にユーザーが混雑を嫌うと言った場合、次回からピーク時の人気観光地を避ける。
3. **エラー修正**：過去に満室のホテルを提案してしまった場合、今後は予約状況をより厳密に確認する。

#### 開発者向け実践例

Travel Agentのコードにメタ認知を組み込む簡単な例は以下の通りです：

```python
class Travel_Agent:
    def __init__(self):
        self.user_preferences = {}
        self.experience_data = []

    def gather_preferences(self, preferences):
        self.user_preferences = preferences

    def retrieve_information(self):
        # Search for flights, hotels, and attractions based on preferences
        flights = search_flights(self.user_preferences)
        hotels = search_hotels(self.user_preferences)
        attractions = search_attractions(self.user_preferences)
        return flights, hotels, attractions

    def generate_recommendations(self):
        flights, hotels, attractions = self.retrieve_information()
        itinerary = create_itinerary(flights, hotels, attractions)
        return itinerary

    def adjust_based_on_feedback(self, feedback):
        self.experience_data.append(feedback)
        # Analyze feedback and adjust future recommendations
        self.user_preferences = adjust_preferences(self.user_preferences, feedback)

# Example usage
travel_agent = Travel_Agent()
preferences = {
    "destination": "Paris",
    "dates": "2025-04-01 to 2025-04-10",
    "budget": "moderate",
    "interests": ["museums", "cuisine"]
}
travel_agent.gather_preferences(preferences)
itinerary = travel_agent.generate_recommendations()
print("Suggested Itinerary:", itinerary)
feedback = {"liked": ["Louvre Museum"], "disliked": ["Eiffel Tower (too crowded)"]}
travel_agent.adjust_based_on_feedback(feedback)
```

#### メタ認知が重要な理由

- **自己反省**：パフォーマンスを分析し、改善点を見つける。
- **適応性**：フィードバックや状況の変化に基づいて戦略を修正する。
- **エラー修正**：誤りを自律的に検出・修正する。
- **リソース管理**：時間や計算リソースの最適化を図る。

メタ認知を取り入れることで、Travel Agentはよりパーソナライズされ正確な旅行提案を行い、ユーザー体験を向上させます。

---

## 2. エージェントにおける計画

計画はAIエージェントの行動における重要な要素です。目標達成のために必要な手順を、現在の状況、利用可能なリソース、考えられる障害を考慮して整理します。

### 計画の要素

- **現在のタスク**：タスクを明確に定義する。
- **タスク完了の手順**：タスクを管理しやすいステップに分解する。
- **必要なリソース**：必要なリソースを特定する。
- **経験**：過去の経験を計画に活かす。

**例**：Travel Agentがユーザーの旅行計画を効果的に支援するための手順は以下の通りです。

### Travel Agentの手順

1. **ユーザーの好みを収集**

   - 旅行日程、予算、興味、特別な要望をユーザーに尋ねる。
   - 例：「旅行予定はいつですか？」「予算の範囲は？」「休暇で楽しみたいことは何ですか？」

2. **情報収集**

   - ユーザーの好みに基づいて関連する旅行オプションを検索する。
   - **フライト**：予算内で希望の日程に合うフライトを探す。
   - **宿泊施設**：場所、価格、設備の好みに合うホテルやレンタル物件を探す。
   - **観光地・レストラン**：ユーザーの興味に合う人気スポットや食事場所を特定する。

3. **提案作成**

   - 収集した情報をもとにパーソナライズされた旅程を作成する。
   - フライト情報、ホテル予約、推奨アクティビティなどを含め、ユーザーの好みに合わせて調整する。

4. **旅程をユーザーに提示**

   - 提案した旅程をユーザーに共有し、レビューしてもらう。
   - 例：「パリ旅行の提案旅程です。フライト詳細、ホテル予約、推奨アクティビティとレストランのリストを含みます。ご意見をお聞かせください。」

5. **フィードバック収集**

   - 提案に対するユーザーの感想を聞く。
   - 例：「フライト案は気に入りましたか？」「ホテルは希望に合っていますか？」「追加や削除したいアクティビティはありますか？」

6. **フィードバックに基づく調整**

   - ユーザーの意見に応じて旅程を修正する。
   - フライト、宿泊、アクティビティの提案を好みに合わせて変更する。

7. **最終確認**

   - 更新した旅程をユーザーに提示し、最終確認を行う。
   - 例：「ご意見を反映しました。こちらが更新済みの旅程です。ご確認ください。」

8. **予約確定**

   - ユーザーの承認を得たら、フライトや宿泊、事前予約したアクティビティの手配を行う。
   - 確認内容をユーザーに送付する。

9. **継続的なサポート提供**

   - 旅行前後や旅行中に変更や追加の要望に対応できるよう待機する。
   - 例：「旅行中に何かあれば、いつでもご連絡ください！」

### 例：やりとりの例

```python
class Travel_Agent:
    def __init__(self):
        self.user_preferences = {}
        self.experience_data = []

    def gather_preferences(self, preferences):
        self.user_preferences = preferences

    def retrieve_information(self):
        flights = search_flights(self.user_preferences)
        hotels = search_hotels(self.user_preferences)
        attractions = search_attractions(self.user_preferences)
        return flights, hotels, attractions

    def generate_recommendations(self):
        flights, hotels, attractions = self.retrieve_information()
        itinerary = create_itinerary(flights, hotels, attractions)
        return itinerary

    def adjust_based_on_feedback(self, feedback):
        self.experience_data.append(feedback)
        self.user_preferences = adjust_preferences(self.user_preferences, feedback)

# Example usage within a booing request
travel_agent = Travel_Agent()
preferences = {
    "destination": "Paris",
    "dates": "2025-04-01 to 2025-04-10",
    "budget": "moderate",
    "interests": ["museums", "cuisine"]
}
travel_agent.gather_preferences(preferences)
itinerary = travel_agent.generate_recommendations()
print("Suggested Itinerary:", itinerary)
feedback = {"liked": ["Louvre Museum"], "disliked": ["Eiffel Tower (too crowded)"]}
travel_agent.adjust_based_on_feedback(feedback)
```

## 3. 修正型RAGシステム

まずはRAGツールと先取りコンテキストロードの違いを理解しましょう。

![RAG vs Context Loading](../../../translated_images/rag-vs-context.9bb2b76d17aeba1489ad2a43ddbc9cd20e7ada4e4871cc99c63a498aa0ff70f7.ja.png)

### Retrieval-Augmented Generation (RAG)

RAGは検索システムと生成モデルを組み合わせたものです。クエリが発生すると、検索システムが外部ソースから関連文書やデータを取得し、その情報を生成モデルへの入力に加えます。これにより、より正確で文脈に沿った応答が生成されます。

RAGシステムでは、エージェントが知識ベースから関連情報を取り出し、それを使って適切な応答や行動を生成します。

### 修正型RAGアプローチ

修正型RAGは、RAG技術を使ってエラーを修正し、AIエージェントの精度を向上させることに焦点を当てています。これには以下が含まれます：

1. **プロンプト技術**：エージェントが関連情報を取得できるように特定のプロンプトを使う。
2. **ツール**：エージェントが取得情報の関連性を評価し、正確な応答を生成するためのアルゴリズムや仕組みを実装する。
3. **評価**：エージェントの性能を継続的に評価し、精度や効率を改善するために調整を行う。
例：検索エージェントにおける修正的RAG  
検索エージェントがウェブから情報を取得してユーザーの質問に答えるとします。修正的RAGのアプローチは以下を含むかもしれません：  
1. **プロンプト技術**：ユーザーの入力に基づいて検索クエリを作成する。  
2. **ツール**：自然言語処理や機械学習アルゴリズムを用いて検索結果をランク付けし、フィルタリングする。  
3. **評価**：ユーザーフィードバックを分析し、取得した情報の誤りを特定して修正する。  

### 旅行エージェントにおける修正的RAG  
修正的RAG（Retrieval-Augmented Generation）は、AIが情報を取得・生成する能力を強化し、誤りを修正します。旅行エージェントが修正的RAGアプローチを使って、より正確かつ関連性の高い旅行提案を行う方法を見てみましょう。  
これには以下が含まれます：  
- **プロンプト技術**：エージェントが関連情報を取得するための特定のプロンプトを使用する。  
- **ツール**：取得情報の関連性を評価し、正確な応答を生成するためのアルゴリズムや仕組みを実装する。  
- **評価**：エージェントのパフォーマンスを継続的に評価し、精度と効率を向上させるために調整を行う。  

#### 旅行エージェントで修正的RAGを実装する手順  
1. **初期ユーザーインタラクション**  
   - 旅行エージェントは、目的地、旅行日程、予算、興味などユーザーの初期の希望を収集する。  
   - 例：```python
     preferences = {
         "destination": "Paris",
         "dates": "2025-04-01 to 2025-04-10",
         "budget": "moderate",
         "interests": ["museums", "cuisine"]
     }
     ```  
2. **情報の取得**  
   - 旅行エージェントはユーザーの希望に基づき、フライト、宿泊施設、観光地、レストランなどの情報を取得する。  
   - 例：```python
     flights = search_flights(preferences)
     hotels = search_hotels(preferences)
     attractions = search_attractions(preferences)
     ```  
3. **初期提案の生成**  
   - 取得した情報を使って、パーソナライズされた旅程を生成する。  
   - 例：```python
     itinerary = create_itinerary(flights, hotels, attractions)
     print("Suggested Itinerary:", itinerary)
     ```  
4. **ユーザーフィードバックの収集**  
   - 旅行エージェントは初期提案に対するユーザーのフィードバックを求める。  
   - 例：```python
     feedback = {
         "liked": ["Louvre Museum"],
         "disliked": ["Eiffel Tower (too crowded)"]
     }
     ```  
5. **修正的RAGプロセス**  
   - **プロンプト技術**：ユーザーフィードバックに基づき新たな検索クエリを作成する。  
   - 例：```python
       if "disliked" in feedback:
           preferences["avoid"] = feedback["disliked"]
       ```  
   - **ツール**：ユーザーフィードバックに基づいて関連性を強調し、新たな検索結果をランク付け・フィルタリングするアルゴリズムを使用する。  
   - 例：```python
       new_attractions = search_attractions(preferences)
       new_itinerary = create_itinerary(flights, hotels, new_attractions)
       print("Updated Itinerary:", new_itinerary)
       ```  
   - **評価**：ユーザーフィードバックを分析し、提案の関連性と正確性を継続的に評価し必要な調整を行う。  
   - 例：```python
       def adjust_preferences(preferences, feedback):
           if "liked" in feedback:
               preferences["favorites"] = feedback["liked"]
           if "disliked" in feedback:
               preferences["avoid"] = feedback["disliked"]
           return preferences

       preferences = adjust_preferences(preferences, feedback)
       ```  

#### 実践例  
旅行エージェントに修正的RAGアプローチを組み込んだ簡単なPythonコード例：  
```python
class Travel_Agent:
    def __init__(self):
        self.user_preferences = {}
        self.experience_data = []

    def gather_preferences(self, preferences):
        self.user_preferences = preferences

    def retrieve_information(self):
        flights = search_flights(self.user_preferences)
        hotels = search_hotels(self.user_preferences)
        attractions = search_attractions(self.user_preferences)
        return flights, hotels, attractions

    def generate_recommendations(self):
        flights, hotels, attractions = self.retrieve_information()
        itinerary = create_itinerary(flights, hotels, attractions)
        return itinerary

    def adjust_based_on_feedback(self, feedback):
        self.experience_data.append(feedback)
        self.user_preferences = adjust_preferences(self.user_preferences, feedback)
        new_itinerary = self.generate_recommendations()
        return new_itinerary

# Example usage
travel_agent = Travel_Agent()
preferences = {
    "destination": "Paris",
    "dates": "2025-04-01 to 2025-04-10",
    "budget": "moderate",
    "interests": ["museums", "cuisine"]
}
travel_agent.gather_preferences(preferences)
itinerary = travel_agent.generate_recommendations()
print("Suggested Itinerary:", itinerary)
feedback = {"liked": ["Louvre Museum"], "disliked": ["Eiffel Tower (too crowded)"]}
new_itinerary = travel_agent.adjust_based_on_feedback(feedback)
print("Updated Itinerary:", new_itinerary)
```  

### 先取りコンテキストロード  
先取りコンテキストロードは、クエリ処理の前に関連コンテキストや背景情報をモデルに読み込むことを指します。これにより、モデルは開始時からこの情報にアクセスでき、処理中に追加データを取得することなく、より情報に基づいた応答を生成できます。  
旅行エージェントアプリケーションでの先取りコンテキストロードの簡単なPython例：  
```python
class TravelAgent:
    def __init__(self):
        # Pre-load popular destinations and their information
        self.context = {
            "Paris": {"country": "France", "currency": "Euro", "language": "French", "attractions": ["Eiffel Tower", "Louvre Museum"]},
            "Tokyo": {"country": "Japan", "currency": "Yen", "language": "Japanese", "attractions": ["Tokyo Tower", "Shibuya Crossing"]},
            "New York": {"country": "USA", "currency": "Dollar", "language": "English", "attractions": ["Statue of Liberty", "Times Square"]},
            "Sydney": {"country": "Australia", "currency": "Dollar", "language": "English", "attractions": ["Sydney Opera House", "Bondi Beach"]}
        }

    def get_destination_info(self, destination):
        # Fetch destination information from pre-loaded context
        info = self.context.get(destination)
        if info:
            return f"{destination}:\nCountry: {info['country']}\nCurrency: {info['currency']}\nLanguage: {info['language']}\nAttractions: {', '.join(info['attractions'])}"
        else:
            return f"Sorry, we don't have information on {destination}."

# Example usage
travel_agent = TravelAgent()
print(travel_agent.get_destination_info("Paris"))
print(travel_agent.get_destination_info("Tokyo"))
```  

#### 説明  
1. **初期化 (`__init__` method)**: The `TravelAgent` class pre-loads a dictionary containing information about popular destinations such as Paris, Tokyo, New York, and Sydney. This dictionary includes details like the country, currency, language, and major attractions for each destination.

2. **Retrieving Information (`get_destination_info` method)**: When a user queries about a specific destination, the `get_destination_info` メソッド)**  
   - 事前にロードされたコンテキスト辞書から関連情報を取得します。  
   - 先取りコンテキストにより、旅行エージェントアプリは外部ソースからリアルタイムに情報を取得することなく、迅速にユーザーの問い合わせに応答可能となり、効率的かつ応答性が向上します。  

### 反復前に目標でプランをブートストラップする  
目標を持ってプランをブートストラップするとは、明確な目的や達成したい結果を最初に設定することです。これにより、モデルは反復プロセス全体を通じてその目標を指針として用いることができ、各反復が望ましい結果に近づくように進み、効率的かつ焦点の合ったプロセスになります。  
旅行エージェントで目標を持ってプランをブートストラップし、その後反復する例（Python）：  

### シナリオ  
旅行エージェントがクライアントのカスタマイズされた休暇を計画したい。目標は、クライアントの好みと予算に基づいて満足度を最大化する旅程を作成すること。  

### 手順  
1. クライアントの好みと予算を定義する。  
2. これらの好みに基づいて初期プランをブートストラップする。  
3. クライアントの満足度を最適化するためにプランを反復的に改良する。  

#### Pythonコード  
```python
class TravelAgent:
    def __init__(self, destinations):
        self.destinations = destinations

    def bootstrap_plan(self, preferences, budget):
        plan = []
        total_cost = 0

        for destination in self.destinations:
            if total_cost + destination['cost'] <= budget and self.match_preferences(destination, preferences):
                plan.append(destination)
                total_cost += destination['cost']

        return plan

    def match_preferences(self, destination, preferences):
        for key, value in preferences.items():
            if destination.get(key) != value:
                return False
        return True

    def iterate_plan(self, plan, preferences, budget):
        for i in range(len(plan)):
            for destination in self.destinations:
                if destination not in plan and self.match_preferences(destination, preferences) and self.calculate_cost(plan, destination) <= budget:
                    plan[i] = destination
                    break
        return plan

    def calculate_cost(self, plan, new_destination):
        return sum(destination['cost'] for destination in plan) + new_destination['cost']

# Example usage
destinations = [
    {"name": "Paris", "cost": 1000, "activity": "sightseeing"},
    {"name": "Tokyo", "cost": 1200, "activity": "shopping"},
    {"name": "New York", "cost": 900, "activity": "sightseeing"},
    {"name": "Sydney", "cost": 1100, "activity": "beach"},
]

preferences = {"activity": "sightseeing"}
budget = 2000

travel_agent = TravelAgent(destinations)
initial_plan = travel_agent.bootstrap_plan(preferences, budget)
print("Initial Plan:", initial_plan)

refined_plan = travel_agent.iterate_plan(initial_plan, preferences, budget)
print("Refined Plan:", refined_plan)
```  

#### コード説明  
1. **初期化 (`__init__` method)**: The `TravelAgent` class is initialized with a list of potential destinations, each having attributes like name, cost, and activity type.

2. **Bootstrapping the Plan (`bootstrap_plan` method)**: This method creates an initial travel plan based on the client's preferences and budget. It iterates through the list of destinations and adds them to the plan if they match the client's preferences and fit within the budget.

3. **Matching Preferences (`match_preferences` method)**: This method checks if a destination matches the client's preferences.

4. **Iterating the Plan (`iterate_plan` method)**: This method refines the initial plan by trying to replace each destination in the plan with a better match, considering the client's preferences and budget constraints.

5. **Calculating Cost (`calculate_cost` メソッド)**  
   - 現在のプランと潜在的な新しい目的地を含む合計コストを計算します。  

#### 使用例  
- **初期プラン**：旅行エージェントは観光の好みと2000ドルの予算に基づいて初期プランを作成。  
- **改良プラン**：クライアントの好みと予算に最適化するためにプランを反復。  
明確な目標（例：クライアント満足度の最大化）でプランをブートストラップし、反復して改良することで、旅行エージェントはクライアントにカスタマイズされ最適化された旅程を作成できます。このアプローチにより、旅行プランは最初からクライアントの好みと予算に合致し、反復ごとに改善されます。  

### LLMを活用した再ランク付けとスコアリング  
大規模言語モデル（LLM）は、取得した文書や生成された応答の関連性と品質を評価することで、再ランク付けやスコアリングに利用できます。仕組みは以下の通りです：  
**取得**：初期取得ステップで、クエリに基づく候補文書や応答のセットを取得。  
**再ランク付け**：LLMがこれらの候補を評価し、関連性と品質に基づいて再ランク付け。これにより、最も関連性が高く質の良い情報が最初に提示される。  
**スコアリング**：LLMは各候補に関連性と品質を反映したスコアを割り当てる。これにより、最適な応答や文書を選択可能。  
LLMを再ランク付けとスコアリングに活用することで、より正確で文脈に合った情報提供が可能になり、ユーザー体験が向上します。  
以下は、旅行エージェントがユーザーの好みに基づいて旅行先を再ランク付け・スコアリングする例（Python）：  

#### シナリオ  
- ユーザーの好みに基づいて最適な旅行先を推薦したい。  
- LLMが旅行先を再ランク付けし、最も関連性の高い選択肢を提示する。  

#### 手順  
1. ユーザーの好みを収集。  
2. 潜在的な旅行先リストを取得。  
3. LLMを用いて旅行先をユーザーの好みに基づき再ランク付けし、スコアリング。  

Azure OpenAIサービスを利用するための更新例：  

#### 要件  
1. Azureサブスクリプションが必要。  
2. Azure OpenAIリソースを作成し、APIキーを取得。  

#### Pythonコード例  
```python
import requests
import json

class TravelAgent:
    def __init__(self, destinations):
        self.destinations = destinations

    def get_recommendations(self, preferences, api_key, endpoint):
        # Generate a prompt for the Azure OpenAI
        prompt = self.generate_prompt(preferences)
        
        # Define headers and payload for the request
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        payload = {
            "prompt": prompt,
            "max_tokens": 150,
            "temperature": 0.7
        }
        
        # Call the Azure OpenAI API to get the re-ranked and scored destinations
        response = requests.post(endpoint, headers=headers, json=payload)
        response_data = response.json()
        
        # Extract and return the recommendations
        recommendations = response_data['choices'][0]['text'].strip().split('\n')
        return recommendations

    def generate_prompt(self, preferences):
        prompt = "Here are the travel destinations ranked and scored based on the following user preferences:\n"
        for key, value in preferences.items():
            prompt += f"{key}: {value}\n"
        prompt += "\nDestinations:\n"
        for destination in self.destinations:
            prompt += f"- {destination['name']}: {destination['description']}\n"
        return prompt

# Example usage
destinations = [
    {"name": "Paris", "description": "City of lights, known for its art, fashion, and culture."},
    {"name": "Tokyo", "description": "Vibrant city, famous for its modernity and traditional temples."},
    {"name": "New York", "description": "The city that never sleeps, with iconic landmarks and diverse culture."},
    {"name": "Sydney", "description": "Beautiful harbour city, known for its opera house and stunning beaches."},
]

preferences = {"activity": "sightseeing", "culture": "diverse"}
api_key = 'your_azure_openai_api_key'
endpoint = 'https://your-endpoint.com/openai/deployments/your-deployment-name/completions?api-version=2022-12-01'

travel_agent = TravelAgent(destinations)
recommendations = travel_agent.get_recommendations(preferences, api_key, endpoint)
print("Recommended Destinations:")
for rec in recommendations:
    print(rec)
```  

#### コード説明  
- Preference Booker  
1. **初期化**：`TravelAgent` class is initialized with a list of potential travel destinations, each having attributes like name and description.

2. **Getting Recommendations (`get_recommendations` method)**: This method generates a prompt for the Azure OpenAI service based on the user's preferences and makes an HTTP POST request to the Azure OpenAI API to get re-ranked and scored destinations.

3. **Generating Prompt (`generate_prompt` method)**: This method constructs a prompt for the Azure OpenAI, including the user's preferences and the list of destinations. The prompt guides the model to re-rank and score the destinations based on the provided preferences.

4. **API Call**: The `requests` library is used to make an HTTP POST request to the Azure OpenAI API endpoint. The response contains the re-ranked and scored destinations.

5. **Example Usage**: The travel agent collects user preferences (e.g., interest in sightseeing and diverse culture) and uses the Azure OpenAI service to get re-ranked and scored recommendations for travel destinations.

Make sure to replace `your_azure_openai_api_key` with your actual Azure OpenAI API key and `https://your-endpoint.com/...` は、Azure OpenAIの実際のエンドポイントURLに置き換える。  
LLMを再ランク付けとスコアリングに活用することで、旅行エージェントはクライアントによりパーソナライズされ関連性の高い旅行提案を提供し、全体的な体験を向上させられます。  

### RAG：プロンプト技術とツールの比較  
Retrieval-Augmented Generation（RAG）は、AIエージェント開発においてプロンプト技術としてもツールとしても利用可能です。両者の違いを理解することで、RAGをより効果的に活用できます。  

#### プロンプト技術としてのRAG  
**何か？**  
- プロンプト技術としてのRAGは、大規模なコーパスやデータベースから関連情報を取得するための特定のクエリやプロンプトを作成することを指します。この情報を使って応答やアクションを生成します。  
**動作方法：**  
1. **プロンプト作成**：タスクやユーザー入力に基づき、構造化されたプロンプトやクエリを作成。  
2. **情報取得**：作成したプロンプトを使い、既存の知識ベースやデータセットから関連データを検索。  
3. **応答生成**：取得した情報と生成AIモデルを組み合わせて、包括的で一貫した応答を生成。  
**旅行エージェントの例**：  
- ユーザー入力：「パリの博物館を訪れたい」  
- プロンプト：「パリのトップ博物館を探してください」  
- 取得情報：ルーブル美術館、オルセー美術館などの詳細  
- 生成応答：「パリのトップ博物館はこちらです：ルーブル美術館、オルセー美術館、ポンピドゥーセンター」  

#### ツールとしてのRAG  
**何か？**  
- ツールとしてのRAGは、取得と生成のプロセスを自動化する統合システムであり、開発者が各クエリに対して手動でプロンプトを作成することなく複雑なAI機能を実装しやすくします。  
**動作方法：**  
1. **統合**：AIエージェントのアーキテクチャにRAGを組み込み、自動的に取得と生成タスクを処理。  
2. **自動化**：ユーザー入力の受け取りから最終応答の生成まで、明示的なプロンプトなしでプロセスを管理。  
3. **効率化**：取得と生成のプロセスを合理化し、迅速かつ正確な応答を可能に。  
**旅行エージェントの例**：  
- ユーザー入力：「パリの博物館を訪れたい」  
- RAGツール：自動的に博物館情報を取得し応答を生成。  
- 生成応答：「パリのトップ博物館はこちらです：ルーブル美術館、オルセー美術館、ポンピドゥーセンター」  

### 比較表  
| 観点 | プロンプト技術 | ツール |  
|------------------------|-------------------------------------------------------------|-------------------------------------------------------|  
| **手動 vs 自動** | 各クエリに対して手動でプロンプト作成 | 取得と生成を自動化 |  
| **制御** | 取得プロセスを細かく制御可能 | 取得と生成を合理化・自動化 |  
| **柔軟性** | 特定ニーズに応じてカスタマイズ可能 | 大規模実装に効率的 |  
| **複雑さ** | プロンプトの作成と調整が必要 | AIエージェントのアーキテクチャに統合しやすい |  

### 実践例  
**プロンプト技術の例：**  
```python
def search_museums_in_paris():
    prompt = "Find top museums in Paris"
    search_results = search_web(prompt)
    return search_results

museums = search_museums_in_paris()
print("Top Museums in Paris:", museums)
```  
**ツールの例：**  
```python
class Travel_Agent:
    def __init__(self):
        self.rag_tool = RAGTool()

    def get_museums_in_paris(self):
        user_input = "I want to visit museums in Paris."
        response = self.rag_tool.retrieve_and_generate(user_input)
        return response

travel_agent = Travel_Agent()
museums = travel_agent.get_museums_in_paris()
print("Top Museums in Paris:", museums)
```  

### 関連性の評価  
関連性の評価は、AIエージェントのパフォーマンスにおいて重要です。取得・生成された情報が適切で正確かつユーザーにとって有用であることを保証します。AIエージェントで関連性を評価する方法を、実践例や技術とともに探ります。  

#### 関連性評価の重要な概念  
1. **コンテキスト認識**：  
   - エージェントはユーザーのクエリの文脈を理解し、関連情報を取得・生成しなければならない。  
   - 例：「パリのおすすめレストラン」と聞かれた場合、料理の種類や予算などユーザーの好みを考慮する。  
2. **正確性**：  
   - エージェントが提供する情報は事実に基づき最新でなければならない。  
   - 例：現在営業中で良い評価のレストランを推奨し、閉店や古い情報は避ける。  
3. **ユーザー意図**：
エージェントは、ユーザーのクエリの背後にある意図を推測し、最も関連性の高い情報を提供すべきです。  
- 例：ユーザーが「予算に優しいホテル」を求めた場合、エージェントは手頃なオプションを優先すべきです。  

4. **フィードバックループ**：  
- ユーザーのフィードバックを継続的に収集・分析することで、エージェントは関連性評価プロセスを改善できます。  
- 例：過去の推薦に対するユーザー評価やフィードバックを取り入れ、将来の応答を改善する。  

#### 関連性評価の実践的手法  
1. **関連性スコアリング**：  
- 取得した各アイテムに対し、ユーザーのクエリや好みにどれだけ合致しているかに基づき関連性スコアを割り当てる。  
- 例：```python
     def relevance_score(item, query):
         score = 0
         if item['category'] in query['interests']:
             score += 1
         if item['price'] <= query['budget']:
             score += 1
         if item['location'] == query['destination']:
             score += 1
         return score
     ```  

2. **フィルタリングとランキング**：  
- 関連性の低いアイテムを除外し、残りを関連性スコアに基づいてランク付けする。  
- 例：```python
     def filter_and_rank(items, query):
         ranked_items = sorted(items, key=lambda item: relevance_score(item, query), reverse=True)
         return ranked_items[:10]  # Return top 10 relevant items
     ```  

3. **自然言語処理（NLP）**：  
- NLP技術を用いてユーザーのクエリを理解し、関連情報を取得する。  
- 例：```python
     def process_query(query):
         # Use NLP to extract key information from the user's query
         processed_query = nlp(query)
         return processed_query
     ```  

4. **ユーザーフィードバック統合**：  
- 提供した推薦に対するユーザーフィードバックを収集し、将来の関連性評価に反映させる。  
- 例：```python
     def adjust_based_on_feedback(feedback, items):
         for item in items:
             if item['name'] in feedback['liked']:
                 item['relevance'] += 1
             if item['name'] in feedback['disliked']:
                 item['relevance'] -= 1
         return items
     ```  

#### 例：Travel Agentにおける関連性評価  
Travel Agentが旅行の推薦の関連性を評価する実践例：```python
class Travel_Agent:
    def __init__(self):
        self.user_preferences = {}
        self.experience_data = []

    def gather_preferences(self, preferences):
        self.user_preferences = preferences

    def retrieve_information(self):
        flights = search_flights(self.user_preferences)
        hotels = search_hotels(self.user_preferences)
        attractions = search_attractions(self.user_preferences)
        return flights, hotels, attractions

    def generate_recommendations(self):
        flights, hotels, attractions = self.retrieve_information()
        ranked_hotels = self.filter_and_rank(hotels, self.user_preferences)
        itinerary = create_itinerary(flights, ranked_hotels, attractions)
        return itinerary

    def filter_and_rank(self, items, query):
        ranked_items = sorted(items, key=lambda item: self.relevance_score(item, query), reverse=True)
        return ranked_items[:10]  # Return top 10 relevant items

    def relevance_score(self, item, query):
        score = 0
        if item['category'] in query['interests']:
            score += 1
        if item['price'] <= query['budget']:
            score += 1
        if item['location'] == query['destination']:
            score += 1
        return score

    def adjust_based_on_feedback(self, feedback, items):
        for item in items:
            if item['name'] in feedback['liked']:
                item['relevance'] += 1
            if item['name'] in feedback['disliked']:
                item['relevance'] -= 1
        return items

# Example usage
travel_agent = Travel_Agent()
preferences = {
    "destination": "Paris",
    "dates": "2025-04-01 to 2025-04-10",
    "budget": "moderate",
    "interests": ["museums", "cuisine"]
}
travel_agent.gather_preferences(preferences)
itinerary = travel_agent.generate_recommendations()
print("Suggested Itinerary:", itinerary)
feedback = {"liked": ["Louvre Museum"], "disliked": ["Eiffel Tower (too crowded)"]}
updated_items = travel_agent.adjust_based_on_feedback(feedback, itinerary['hotels'])
print("Updated Itinerary with Feedback:", updated_items)
```  

### 意図を持った検索  
意図を持った検索とは、ユーザーのクエリの根底にある目的やゴールを理解・解釈し、最も関連性が高く有用な情報を取得・生成することです。単にキーワードをマッチさせるだけでなく、ユーザーの実際のニーズや文脈を把握することに重点を置いています。  

#### 意図を持った検索の主要概念  
1. **ユーザーの意図の理解**：  
- ユーザーの意図は主に情報収集型、ナビゲーション型、トランザクション型の3種類に分類される。  
- **情報収集型意図**：ユーザーが特定のトピックに関する情報を求めている（例：「パリのおすすめ博物館は？」）。  
- **ナビゲーション型意図**：ユーザーが特定のウェブサイトやページにアクセスしたい（例：「ルーブル美術館公式サイト」）。  
- **トランザクション型意図**：ユーザーが予約や購入などの取引を行いたい（例：「パリ行きのフライトを予約」）。  

2. **コンテキスト認識**：  
- ユーザーのクエリの文脈を分析し、意図を正確に特定する。過去のやり取り、ユーザーの好み、現在のクエリの詳細を考慮する。  

3. **自然言語処理（NLP）**：  
- NLP技術を用いて、ユーザーが入力した自然言語のクエリを理解・解釈する。エンティティ認識、感情分析、クエリ解析などのタスクを含む。  

4. **パーソナライゼーション**：  
- ユーザーの履歴、好み、フィードバックに基づき検索結果を個別化し、取得情報の関連性を高める。  

#### 実践例：Travel Agentにおける意図を持った検索  
Travel Agentを例に、意図を持った検索の実装例を示す。  
1. **ユーザーの好み収集** ```python
   class Travel_Agent:
       def __init__(self):
           self.user_preferences = {}

       def gather_preferences(self, preferences):
           self.user_preferences = preferences
   ```  
2. **ユーザーの意図理解** ```python
   def identify_intent(query):
       if "book" in query or "purchase" in query:
           return "transactional"
       elif "website" in query or "official" in query:
           return "navigational"
       else:
           return "informational"
   ```  
3. **コンテキスト認識** ```python
   def analyze_context(query, user_history):
       # Combine current query with user history to understand context
       context = {
           "current_query": query,
           "user_history": user_history
       }
       return context
   ```  
4. **検索と結果のパーソナライズ** ```python
   def search_with_intent(query, preferences, user_history):
       intent = identify_intent(query)
       context = analyze_context(query, user_history)
       if intent == "informational":
           search_results = search_information(query, preferences)
       elif intent == "navigational":
           search_results = search_navigation(query)
       elif intent == "transactional":
           search_results = search_transaction(query, preferences)
       personalized_results = personalize_results(search_results, user_history)
       return personalized_results

   def search_information(query, preferences):
       # Example search logic for informational intent
       results = search_web(f"best {preferences['interests']} in {preferences['destination']}")
       return results

   def search_navigation(query):
       # Example search logic for navigational intent
       results = search_web(query)
       return results

   def search_transaction(query, preferences):
       # Example search logic for transactional intent
       results = search_web(f"book {query} to {preferences['destination']}")
       return results

   def personalize_results(results, user_history):
       # Example personalization logic
       personalized = [result for result in results if result not in user_history]
       return personalized[:10]  # Return top 10 personalized results
   ```  
5. **使用例** ```python
   travel_agent = Travel_Agent()
   preferences = {
       "destination": "Paris",
       "interests": ["museums", "cuisine"]
   }
   travel_agent.gather_preferences(preferences)
   user_history = ["Louvre Museum website", "Book flight to Paris"]
   query = "best museums in Paris"
   results = search_with_intent(query, preferences, user_history)
   print("Search Results:", results)
   ```  

---  

## 4. ツールとしてのコード生成  
コード生成エージェントはAIモデルを用いてコードを書き、実行し、複雑な問題を解決しタスクを自動化します。  

### コード生成エージェント  
コード生成エージェントは生成AIモデルを使い、様々なプログラミング言語でコードを書き実行します。これにより複雑な問題の解決、タスクの自動化、価値ある洞察の提供が可能です。  

#### 実用例  
1. **自動コード生成**：データ分析、ウェブスクレイピング、機械学習など特定タスク向けのコードスニペット生成。  
2. **RAGとしてのSQL**：データベースからのデータ取得や操作にSQLクエリを使用。  
3. **問題解決**：アルゴリズム最適化やデータ分析など特定課題を解決するコードの作成・実行。  

#### 例：データ分析向けコード生成エージェント  
コード生成エージェントの設計例：  
1. **タスク**：データセットを分析し傾向やパターンを特定。  
2. **ステップ**：  
- データセットを分析ツールに読み込む。  
- データをフィルタ・集計するSQLクエリを生成。  
- クエリを実行し結果を取得。  
- 結果を基に可視化や洞察を生成。  
3. **必要リソース**：データセット、分析ツール、SQL機能へのアクセス。  
4. **経験**：過去の分析結果を活用し精度や関連性を向上。  

### 例：Travel Agent向けコード生成エージェント  
Travel Agentというコード生成エージェントを設計し、ユーザーの旅行計画を支援する例。飛行機、ホテル、観光情報の取得、結果のフィルタリング、旅程作成を生成AIで実行。  

#### コード生成エージェントの概要  
1. **ユーザーの好み収集**：目的地、旅行日程、予算、興味などを収集。  
2. **データ取得用コード生成**：フライト、ホテル、観光地情報を取得するコードスニペットを生成。  
3. **生成コードの実行**：リアルタイム情報を取得するためにコードを実行。  
4. **旅程生成**：取得データをまとめてパーソナライズされた旅程を作成。  
5. **フィードバックに基づく調整**：ユーザーフィードバックを受け、必要に応じてコードを再生成し結果を改善。  

#### ステップバイステップ実装  
1. **ユーザーの好み収集** ```python
   class Travel_Agent:
       def __init__(self):
           self.user_preferences = {}

       def gather_preferences(self, preferences):
           self.user_preferences = preferences
   ```  
2. **データ取得用コード生成** ```python
   def generate_code_to_fetch_data(preferences):
       # Example: Generate code to search for flights based on user preferences
       code = f"""
       def search_flights():
           import requests
           response = requests.get('https://api.example.com/flights', params={preferences})
           return response.json()
       """
       return code

   def generate_code_to_fetch_hotels(preferences):
       # Example: Generate code to search for hotels
       code = f"""
       def search_hotels():
           import requests
           response = requests.get('https://api.example.com/hotels', params={preferences})
           return response.json()
       """
       return code
   ```  
3. **生成コードの実行** ```python
   def execute_code(code):
       # Execute the generated code using exec
       exec(code)
       result = locals()
       return result

   travel_agent = Travel_Agent()
   preferences = {
       "destination": "Paris",
       "dates": "2025-04-01 to 2025-04-10",
       "budget": "moderate",
       "interests": ["museums", "cuisine"]
   }
   travel_agent.gather_preferences(preferences)
   
   flight_code = generate_code_to_fetch_data(preferences)
   hotel_code = generate_code_to_fetch_hotels(preferences)
   
   flights = execute_code(flight_code)
   hotels = execute_code(hotel_code)

   print("Flight Options:", flights)
   print("Hotel Options:", hotels)
   ```  
4. **旅程生成** ```python
   def generate_itinerary(flights, hotels, attractions):
       itinerary = {
           "flights": flights,
           "hotels": hotels,
           "attractions": attractions
       }
       return itinerary

   attractions = search_attractions(preferences)
   itinerary = generate_itinerary(flights, hotels, attractions)
   print("Suggested Itinerary:", itinerary)
   ```  
5. **フィードバックに基づく調整** ```python
   def adjust_based_on_feedback(feedback, preferences):
       # Adjust preferences based on user feedback
       if "liked" in feedback:
           preferences["favorites"] = feedback["liked"]
       if "disliked" in feedback:
           preferences["avoid"] = feedback["disliked"]
       return preferences

   feedback = {"liked": ["Louvre Museum"], "disliked": ["Eiffel Tower (too crowded)"]}
   updated_preferences = adjust_based_on_feedback(feedback, preferences)
   
   # Regenerate and execute code with updated preferences
   updated_flight_code = generate_code_to_fetch_data(updated_preferences)
   updated_hotel_code = generate_code_to_fetch_hotels(updated_preferences)
   
   updated_flights = execute_code(updated_flight_code)
   updated_hotels = execute_code(updated_hotel_code)
   
   updated_itinerary = generate_itinerary(updated_flights, updated_hotels, attractions)
   print("Updated Itinerary:", updated_itinerary)
   ```  

### 環境認識と推論の活用  
テーブルのスキーマを活用することで、環境認識と推論に基づいたクエリ生成プロセスを強化できる。例：  
1. **スキーマの理解**：システムがテーブルのスキーマを理解し、クエリ生成の基盤とする。  
2. **フィードバックに基づく調整**：フィードバックに応じてユーザーの好みを調整し、スキーマのどのフィールドを更新すべきか推論。  
3. **クエリの生成と実行**：新しい好みに基づいてフライト・ホテル情報を取得するクエリを生成・実行。  

以下はこれらを組み込んだPythonコード例：```python
def adjust_based_on_feedback(feedback, preferences, schema):
    # Adjust preferences based on user feedback
    if "liked" in feedback:
        preferences["favorites"] = feedback["liked"]
    if "disliked" in feedback:
        preferences["avoid"] = feedback["disliked"]
    # Reasoning based on schema to adjust other related preferences
    for field in schema:
        if field in preferences:
            preferences[field] = adjust_based_on_environment(feedback, field, schema)
    return preferences

def adjust_based_on_environment(feedback, field, schema):
    # Custom logic to adjust preferences based on schema and feedback
    if field in feedback["liked"]:
        return schema[field]["positive_adjustment"]
    elif field in feedback["disliked"]:
        return schema[field]["negative_adjustment"]
    return schema[field]["default"]

def generate_code_to_fetch_data(preferences):
    # Generate code to fetch flight data based on updated preferences
    return f"fetch_flights(preferences={preferences})"

def generate_code_to_fetch_hotels(preferences):
    # Generate code to fetch hotel data based on updated preferences
    return f"fetch_hotels(preferences={preferences})"

def execute_code(code):
    # Simulate execution of code and return mock data
    return {"data": f"Executed: {code}"}

def generate_itinerary(flights, hotels, attractions):
    # Generate itinerary based on flights, hotels, and attractions
    return {"flights": flights, "hotels": hotels, "attractions": attractions}

# Example schema
schema = {
    "favorites": {"positive_adjustment": "increase", "negative_adjustment": "decrease", "default": "neutral"},
    "avoid": {"positive_adjustment": "decrease", "negative_adjustment": "increase", "default": "neutral"}
}

# Example usage
preferences = {"favorites": "sightseeing", "avoid": "crowded places"}
feedback = {"liked": ["Louvre Museum"], "disliked": ["Eiffel Tower (too crowded)"]}
updated_preferences = adjust_based_on_feedback(feedback, preferences, schema)

# Regenerate and execute code with updated preferences
updated_flight_code = generate_code_to_fetch_data(updated_preferences)
updated_hotel_code = generate_code_to_fetch_hotels(updated_preferences)

updated_flights = execute_code(updated_flight_code)
updated_hotels = execute_code(updated_hotel_code)

updated_itinerary = generate_itinerary(updated_flights, updated_hotels, feedback["liked"])
print("Updated Itinerary:", updated_itinerary)
```  

#### 説明 - フィードバックに基づく予約  
1. **スキーマ認識**：`schema` dictionary defines how preferences should be adjusted based on feedback. It includes fields like `favorites` and `avoid`, with corresponding adjustments.
2. **Adjusting Preferences (`adjust_based_on_feedback` method)**: This method adjusts preferences based on user feedback and the schema.
3. **Environment-Based Adjustments (`adjust_based_on_environment` メソッドにより、スキーマとフィードバックに基づいた調整を行う。  
4. **クエリ生成と実行**：調整された好みに基づき更新されたフライト・ホテル情報を取得するコードを生成し、クエリ実行をシミュレート。  
5. **旅程生成**：新しいフライト、ホテル、観光情報に基づいた更新済み旅程を作成。  

環境認識とスキーマに基づく推論を組み込むことで、より正確で関連性の高いクエリが生成され、より良い旅行推薦とパーソナライズされたユーザー体験が実現する。  

### RAG技術としてのSQLの活用  
SQL（構造化問い合わせ言語）はデータベースと対話する強力なツール。RAG（Retrieval-Augmented Generation）アプローチの一部として使うと、データベースから関連データを取得し、AIエージェントの応答やアクションの生成に役立てられる。Travel Agentの文脈でSQLをRAG技術として活用する方法を見ていく。  

#### 主要概念  
1. **データベース操作**：  
- SQLを使いデータベースに問い合わせ、関連情報を取得・操作する。  
- 例：旅行データベースからフライト詳細、ホテル情報、観光地を取得。  

2. **RAGとの統合**：  
- ユーザー入力や好みに基づいてSQLクエリを生成。  
- 取得データを用いてパーソナライズされた推薦やアクションを生成。  

3. **動的クエリ生成**：  
- AIエージェントが文脈やユーザーのニーズに基づき動的にSQLクエリを生成。  
- 例：予算、日程、興味に応じてSQLクエリをカスタマイズ。  

#### 応用例  
- **自動コード生成**：特定タスク向けコードスニペット生成。  
- **RAGとしてのSQL**：データ操作にSQLクエリを使用。  
- **問題解決**：問題解決のためのコード生成・実行。  

**例**：データ分析エージェント  
1. **タスク**：データセットの傾向分析。  
2. **ステップ**：  
- データセットを読み込む。  
- データをフィルタリングするSQLクエリを生成。  
- クエリを実行し結果を取得。  
- 可視化や洞察を生成。  
3. **リソース**：データセットアクセス、SQL機能。  
4. **経験**：過去結果を活用し将来の分析を改善。  

#### 実践例：Travel AgentでのSQL利用  
1. **ユーザーの好み収集** ```python
   class Travel_Agent:
       def __init__(self):
           self.user_preferences = {}

       def gather_preferences(self, preferences):
           self.user_preferences = preferences
   ```  
2. **SQLクエリ生成** ```python
   def generate_sql_query(table, preferences):
       query = f"SELECT * FROM {table} WHERE "
       conditions = []
       for key, value in preferences.items():
           conditions.append(f"{key}='{value}'")
       query += " AND ".join(conditions)
       return query
   ```  
3. **SQLクエリ実行** ```python
   import sqlite3

   def execute_sql_query(query, database="travel.db"):
       connection = sqlite3.connect(database)
       cursor = connection.cursor()
       cursor.execute(query)
       results = cursor.fetchall()
       connection.close()
       return results
   ```  
4. **推薦生成** ```python
   def generate_recommendations(preferences):
       flight_query = generate_sql_query("flights", preferences)
       hotel_query = generate_sql_query("hotels", preferences)
       attraction_query = generate_sql_query("attractions", preferences)
       
       flights = execute_sql_query(flight_query)
       hotels = execute_sql_query(hotel_query)
       attractions = execute_sql_query(attraction_query)
       
       itinerary = {
           "flights": flights,
           "hotels": hotels,
           "attractions": attractions
       }
       return itinerary

   travel_agent = Travel_Agent()
   preferences = {
       "destination": "Paris",
       "dates": "2025-04-01 to 2025-04-10",
       "budget": "moderate",
       "interests": ["museums", "cuisine"]
   }
   travel_agent.gather_preferences(preferences)
   itinerary = generate_recommendations(preferences)
   print("Suggested Itinerary:", itinerary)
   ```  

#### SQLクエリ例  
1. **フライトクエリ** ```sql
   SELECT * FROM flights WHERE destination='Paris' AND dates='2025-04-01 to 2025-04-10' AND budget='moderate';
   ```  
2. **ホテルクエリ** ```sql
   SELECT * FROM hotels WHERE destination='Paris' AND budget='moderate';
   ```  
3. **観光地クエリ** ```sql
   SELECT * FROM attractions WHERE destination='Paris' AND interests='museums, cuisine';
   ```  

SQLをRAG技術の一部として活用することで、Travel AgentのようなAIエージェントは関連データを動的に取得・利用し、正確かつパーソナライズされた推薦を提供できる。  

### メタ認知の例  
メタ認知の実装例として、問題解決中に*意思決定プロセスを振り返る*シンプルなエージェントを作成する。ここでは、エージェントがホテル選択を最適化しようとするが、自身の推論を評価し、誤りや最適でない選択があれば戦略を調整するシステムを構築する。価格と品質の組み合わせでホテルを選ぶ基本例をシミュレートし、決定を「振り返り」調整する。  

#### メタ認知の説明  
1. **初期決定**：エージェントは品質を理解せず最安ホテルを選択。  
2. **振り返りと評価**：初期選択後、ユーザーフィードバックを用いてホテルが「悪い」選択か評価。品質が低すぎれば推論を振り返る。  
3. **戦略調整**：振り返りに基づき戦略を「最安」から「最高品質」へ切り替え、以降の意思決定を改善。  

例：```python
class HotelRecommendationAgent:
    def __init__(self):
        self.previous_choices = []  # Stores the hotels chosen previously
        self.corrected_choices = []  # Stores the corrected choices
        self.recommendation_strategies = ['cheapest', 'highest_quality']  # Available strategies

    def recommend_hotel(self, hotels, strategy):
        """
        Recommend a hotel based on the chosen strategy.
        The strategy can either be 'cheapest' or 'highest_quality'.
        """
        if strategy == 'cheapest':
            recommended = min(hotels, key=lambda x: x['price'])
        elif strategy == 'highest_quality':
            recommended = max(hotels, key=lambda x: x['quality'])
        else:
            recommended = None
        self.previous_choices.append((strategy, recommended))
        return recommended

    def reflect_on_choice(self):
        """
        Reflect on the last choice made and decide if the agent should adjust its strategy.
        The agent considers if the previous choice led to a poor outcome.
        """
        if not self.previous_choices:
            return "No choices made yet."

        last_choice_strategy, last_choice = self.previous_choices[-1]
        # Let's assume we have some user feedback that tells us whether the last choice was good or not
        user_feedback = self.get_user_feedback(last_choice)

        if user_feedback == "bad":
            # Adjust strategy if the previous choice was unsatisfactory
            new_strategy = 'highest_quality' if last_choice_strategy == 'cheapest' else 'cheapest'
            self.corrected_choices.append((new_strategy, last_choice))
            return f"Reflecting on choice. Adjusting strategy to {new_strategy}."
        else:
            return "The choice was good. No need to adjust."

    def get_user_feedback(self, hotel):
        """
        Simulate user feedback based on hotel attributes.
        For simplicity, assume if the hotel is too cheap, the feedback is "bad".
        If the hotel has quality less than 7, feedback is "bad".
        """
        if hotel['price'] < 100 or hotel['quality'] < 7:
            return "bad"
        return "good"

# Simulate a list of hotels (price and quality)
hotels = [
    {'name': 'Budget Inn', 'price': 80, 'quality': 6},
    {'name': 'Comfort Suites', 'price': 120, 'quality': 8},
    {'name': 'Luxury Stay', 'price': 200, 'quality': 9}
]

# Create an agent
agent = HotelRecommendationAgent()

# Step 1: The agent recommends a hotel using the "cheapest" strategy
recommended_hotel = agent.recommend_hotel(hotels, 'cheapest')
print(f"Recommended hotel (cheapest): {recommended_hotel['name']}")

# Step 2: The agent reflects on the choice and adjusts strategy if necessary
reflection_result = agent.reflect_on_choice()
print(reflection_result)

# Step 3: The agent recommends again, this time using the adjusted strategy
adjusted_recommendation = agent.recommend_hotel(hotels, 'highest_quality')
print(f"Adjusted hotel recommendation (highest_quality): {adjusted_recommendation['name']}")
```  

#### エージェントのメタ認知能力  
エージェントは以下を実行可能：  
- 過去の選択と意思決定プロセスを評価。  
- その振り返りに基づき戦略を調整（メタ認知の実践）。  

これは内部フィードバックに基づき推論プロセスを調整できる単純なメタ認知の形態。  

### 結論  
メタ認知はAIエージェントの能力を大幅に向上させる強力なツールである。メタ認知を組み込むことで、
プロセスを通じて、より知的で適応性が高く効率的なエージェントを設計することができます。追加のリソースを活用して、AIエージェントにおけるメタ認知の魅力的な世界をさらに探求しましょう。  
## 前のレッスン  
[マルチエージェントデザインパターン](../08-multi-agent/README.md)  
## 次のレッスン  
[AIエージェントの実運用](../10-ai-agents-production/README.md)

**免責事項**:  
本書類はAI翻訳サービス「Co-op Translator」（https://github.com/Azure/co-op-translator）を使用して翻訳されました。正確性の向上に努めておりますが、自動翻訳には誤りや不正確な部分が含まれる可能性があります。原文の言語で記載されたオリジナルの文書が正式な情報源とみなされるべきです。重要な情報については、専門の人間による翻訳を推奨します。本翻訳の利用により生じたいかなる誤解や誤訳についても、当方は一切責任を負いません。