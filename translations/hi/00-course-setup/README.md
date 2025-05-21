<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "49fa13c21a816ffcc7740ab17ba024a9",
  "translation_date": "2025-05-20T09:40:51+00:00",
  "source_file": "00-course-setup/README.md",
  "language_code": "hi"
}
-->
आपके पास अब इस कोर्स का अपना फोर्क किया हुआ संस्करण निम्न लिंक में होना चाहिए:

![Forked Repo](../../../translated_images/forked-repo.eea246a73044cc984a1e462349e36e7336204f00785e3187b7399905feeada07.hi.png)

## कोड चलाना

यह कोर्स एक श्रृंखला प्रदान करता है Jupyter Notebooks की जिन्हें आप चलाकर AI एजेंट्स बनाने का व्यावहारिक अनुभव प्राप्त कर सकते हैं।

कोड के नमूने निम्न में से किसी एक का उपयोग करते हैं:

**GitHub अकाउंट आवश्यक - मुफ्त**:

1) Semantic Kernel Agent Framework + GitHub Models Marketplace। इसे (semantic-kernel.ipynb) के रूप में लेबल किया गया है।
2) AutoGen Framework + GitHub Models Marketplace। इसे (autogen.ipynb) के रूप में लेबल किया गया है।

**Azure सब्सक्रिप्शन आवश्यक**:
3) Azure AI Foundry + Azure AI Agent Service। इसे (azureaiagent.ipynb) के रूप में लेबल किया गया है।

हम आपको प्रोत्साहित करते हैं कि आप तीनों प्रकार के उदाहरणों को आजमाएं ताकि आप देख सकें कौन सा आपके लिए सबसे अच्छा काम करता है।

आप जो भी विकल्प चुनते हैं, उससे यह तय होगा कि आपको नीचे दिए गए कौन से सेटअप स्टेप्स फॉलो करने हैं:

## आवश्यकताएँ

- Python 3.12+
- GitHub अकाउंट - GitHub Models Marketplace तक पहुंच के लिए
- Azure सब्सक्रिप्शन - Azure AI Foundry तक पहुंच के लिए
- Azure AI Foundry अकाउंट - Azure AI Agent Service तक पहुंच के लिए

हमने इस रिपॉजिटरी की रूट में एक `requirements.txt` फाइल शामिल की है जिसमें कोड नमूनों को चलाने के लिए आवश्यक सभी Python पैकेज मौजूद हैं।

आप इन्हें रिपॉजिटरी की रूट डायरेक्टरी में टर्मिनल पर निम्न कमांड चलाकर इंस्टॉल कर सकते हैं:

```bash
pip install -r requirements.txt
```
हम सलाह देते हैं कि किसी भी टकराव और समस्याओं से बचने के लिए Python वर्चुअल एनवायरनमेंट बनाएं।

## GitHub Models का उपयोग करने वाले नमूनों के लिए सेटअप

### स्टेप 1: अपना GitHub Personal Access Token (PAT) प्राप्त करें

इस समय, यह कोर्स GitHub Models Marketplace का उपयोग करता है जो Large Language Models (LLMs) तक मुफ्त पहुंच प्रदान करता है, जिनका उपयोग AI एजेंट्स बनाने के लिए किया जाएगा।

इस सेवा तक पहुंचने के लिए, आपको GitHub Personal Access Token बनाना होगा।

यह आपके GitHub अकाउंट में जाकर किया जा सकता है।

`Fine-grained tokens` चुनें और फिर `Generate new token` पर क्लिक करें।

`.env` फाइल बनाने के लिए टर्मिनल में निम्न कमांड चलाएं:

```bash
cp .env.example .env
```

यह उदाहरण फाइल की कॉपी बनाएगा और `.env` फाइल बनाएगा।

`.env` फाइल में आपको `GITHUB_TOKEN` वैरिएबल में अपना Personal Access Token डालना होगा।

यदि आप Azure AI Foundry का उपयोग कर रहे हैं, तो टर्मिनल में निम्न कमांड चलाएं:

```bash
cp .env.example .env
```

यह आपको Azure में लॉगिन करने में मदद करेगा।

इसके बाद, आपको अपनी `.env` फाइल में Azure से संबंधित निम्न वैरिएबल्स सेट करनी होंगी:

- `PROJECT_CONNECTION_STRING`
- `AZURE_SUBSCRIPTION_ID`
- `AZURE_AI_PROJECT_NAME`
- `AZURE_OPENAI_SERVICE`
- `AZURE_OPENAI_RESOURCE_GROUP`
- `GLOBAL_LLM_SERVICE`
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME` (जैसे `text-embedding-ada-002`)
- `AZURE_OPENAI_CHAT_DEPLOYMENT_NAME` (जैसे `gpt-4o-mini`)
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `AZURE_SEARCH_SERVICE_ENDPOINT`
- `AZURE_SEARCH_API_KEY`
- `AZURE_OPENAI_API_VERSION`

आप क्रेडेंशियल प्राप्त करने के लिए `DefaultAzureCredential` फ़ंक्शन का उपयोग कर सकते हैं।

```python
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
```

## कहीं फंस गए?

यदि इस सेटअप को चलाने में आपको कोई समस्या आती है, तो हमारे साथ जुड़ें

या

.

## अगला पाठ

अब आप इस कोर्स का कोड चलाने के लिए तैयार हैं, AI एजेंट्स की दुनिया के बारे में और सीखने के लिए शुभकामनाएं!

[AI एजेंट्स और उनके उपयोग के परिचय](../01-intro-to-ai-agents/README.md)

**अस्वीकरण**:  
यह दस्तावेज़ AI अनुवाद सेवा [Co-op Translator](https://github.com/Azure/co-op-translator) का उपयोग करके अनुवादित किया गया है। जबकि हम सटीकता के लिए प्रयासरत हैं, कृपया ध्यान दें कि स्वचालित अनुवादों में त्रुटियाँ या गलतियाँ हो सकती हैं। मूल दस्तावेज़ को उसकी मूल भाषा में प्रामाणिक स्रोत माना जाना चाहिए। महत्वपूर्ण जानकारी के लिए, पेशेवर मानव अनुवाद की सलाह दी जाती है। इस अनुवाद के उपयोग से उत्पन्न किसी भी गलतफहमी या गलत व्याख्या के लिए हम जिम्मेदार नहीं हैं।