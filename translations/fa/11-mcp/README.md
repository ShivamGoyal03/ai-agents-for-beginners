<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9320dd53c82869fd44935d1581eaf7bb",
  "translation_date": "2025-05-21T08:13:26+00:00",
  "source_file": "11-mcp/README.md",
  "language_code": "fa"
}
-->
# درس ۱۱: یکپارچه‌سازی پروتکل مدل کانتکست (MCP)

## اهداف یادگیری
- درک MCP و نقش آن در توسعه عامل‌های هوش مصنوعی  
- راه‌اندازی و پیکربندی سرور MCP برای یکپارچه‌سازی با GitHub  
- ساخت سیستم چندعاملی با استفاده از ابزارهای MCP  
- پیاده‌سازی RAG (تولید تقویت‌شده با بازیابی) با Azure Cognitive Search  

## پیش‌نیازها
- پایتون ۳.۸ به بالا  
- Node.js نسخه ۱۴ به بالا  
- اشتراک Azure  
- حساب GitHub  
- درک پایه‌ای از Semantic Kernel  

## دستورالعمل‌های راه‌اندازی

1. **راه‌اندازی محیط**  
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **پیکربندی سرویس‌های Azure**  
   - ایجاد منبع Azure Cognitive Search  
   - راه‌اندازی سرویس Azure OpenAI  
   - تنظیم متغیرهای محیطی در فایل `.env`  

3. **راه‌اندازی سرور MCP**  
   ```bash
   npm install -g @modelcontextprotocol/server-github
   ```

## ساختار پروژه

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

## اجزای اصلی

### ۱. سیستم چندعاملی  
- عامل GitHub: تحلیل مخزن  
- عامل Hackathon: پیشنهاد پروژه‌ها  
- عامل Events: پیشنهاد رویدادهای فناوری  

### ۲. یکپارچه‌سازی Azure  
- جستجوی شناختی برای ایندکس‌گذاری رویدادها  
- Azure OpenAI برای هوشمندی عامل‌ها  
- پیاده‌سازی الگوی RAG  

### ۳. ابزارهای MCP  
- تحلیل مخزن GitHub  
- بازرسی کد  
- استخراج فراداده  

## مرور کد

نمونه نشان می‌دهد:  
۱. یکپارچه‌سازی سرور MCP  
۲. هماهنگی چندعاملی  
۳. یکپارچه‌سازی Azure Cognitive Search  
۴. پیاده‌سازی الگوی RAG  

ویژگی‌های کلیدی:  
- تحلیل بلادرنگ مخازن GitHub  
- پیشنهادات هوشمند پروژه  
- تطبیق رویدادها با استفاده از Azure Search  
- پاسخ‌های جریانی با Chainlit  

## اجرای نمونه

۱. سرور MCP را راه‌اندازی کنید:  
   ```bash
   npx @modelcontextprotocol/server-github
   ```

۲. برنامه را اجرا کنید:  
   ```bash
   chainlit run app.py -w
   ```

۳. یکپارچه‌سازی را تست کنید:  
   ```
   Example query: "Analyze repositories for username: <github_username>"
   ```

## رفع اشکال

مشکلات رایج و راه‌حل‌ها:  
۱. مشکلات اتصال MCP  
   - اطمینان از اجرای سرور  
   - بررسی در دسترس بودن پورت  
   - تایید توکن‌های GitHub  

۲. مشکلات Azure Search  
   - اعتبارسنجی رشته‌های اتصال  
   - بررسی وجود ایندکس  
   - تایید بارگذاری اسناد  

## گام‌های بعدی
- کاوش ابزارهای بیشتر MCP  
- پیاده‌سازی عامل‌های سفارشی  
- بهبود قابلیت‌های RAG  
- افزودن منابع رویداد بیشتر  

## منابع  
- [MCP for Beginners](https://aka.ms/mcp-for-beginners)  
- [MCP Documentation](https://github.com/microsoft/semantic-kernel/tree/main/python/semantic-kernel/semantic_kernel/connectors/mcp)  
- [Azure Cognitive Search Docs](https://learn.microsoft.com/azure/search/)  
- [Semantic Kernel Guides](https://learn.microsoft.com/semantic-kernel/)

**سلب مسئولیت**:  
این سند با استفاده از سرویس ترجمه هوش مصنوعی [Co-op Translator](https://github.com/Azure/co-op-translator) ترجمه شده است. در حالی که ما برای دقت تلاش می‌کنیم، لطفاً توجه داشته باشید که ترجمه‌های خودکار ممکن است حاوی خطاها یا نادرستی‌هایی باشند. سند اصلی به زبان بومی خود باید به عنوان منبع معتبر در نظر گرفته شود. برای اطلاعات حیاتی، ترجمه حرفه‌ای انسانی توصیه می‌شود. ما مسئول هیچ گونه سوءتفاهم یا تفسیر نادرستی که ناشی از استفاده از این ترجمه باشد، نیستیم.