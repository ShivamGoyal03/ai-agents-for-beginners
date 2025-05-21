<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9320dd53c82869fd44935d1581eaf7bb",
  "translation_date": "2025-05-21T08:12:41+00:00",
  "source_file": "11-mcp/README.md",
  "language_code": "es"
}
-->
# Lección 11: Integración del Protocolo de Contexto del Modelo (MCP)

## Objetivos de aprendizaje
- Comprender qué es MCP y su papel en el desarrollo de agentes de IA
- Configurar y poner en marcha un servidor MCP para la integración con GitHub
- Construir un sistema multiagente utilizando las herramientas MCP
- Implementar RAG (Generación Aumentada por Recuperación) con Azure Cognitive Search

## Requisitos previos
- Python 3.8+
- Node.js 14+
- Suscripción a Azure
- Cuenta de GitHub
- Conocimientos básicos de Semantic Kernel

## Instrucciones de configuración

1. **Configuración del entorno**  
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configurar servicios de Azure**  
   - Crear un recurso de Azure Cognitive Search  
   - Configurar el servicio Azure OpenAI  
   - Establecer las variables de entorno en `.env`

3. **Configuración del servidor MCP**  
   ```bash
   npm install -g @modelcontextprotocol/server-github
   ```

## Estructura del proyecto

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

## Componentes principales

### 1. Sistema multiagente
- Agente GitHub: Análisis de repositorios  
- Agente Hackathon: Recomendaciones de proyectos  
- Agente de eventos: Sugerencias de eventos tecnológicos

### 2. Integración con Azure
- Cognitive Search para indexación de eventos  
- Azure OpenAI para inteligencia del agente  
- Implementación del patrón RAG

### 3. Herramientas MCP
- Análisis de repositorios en GitHub  
- Inspección de código  
- Extracción de metadatos

## Recorrido por el código

El ejemplo muestra:  
1. Integración del servidor MCP  
2. Orquestación multiagente  
3. Integración con Azure Cognitive Search  
4. Implementación del patrón RAG

Características clave:  
- Análisis en tiempo real de repositorios GitHub  
- Recomendaciones inteligentes de proyectos  
- Coincidencia de eventos usando Azure Search  
- Respuestas en streaming con Chainlit

## Ejecución del ejemplo

1. Iniciar el servidor MCP:  
   ```bash
   npx @modelcontextprotocol/server-github
   ```

2. Lanzar la aplicación:  
   ```bash
   chainlit run app.py -w
   ```

3. Probar la integración:  
   ```
   Example query: "Analyze repositories for username: <github_username>"
   ```

## Solución de problemas

Problemas comunes y soluciones:  
1. Problemas de conexión MCP  
   - Verificar que el servidor esté en funcionamiento  
   - Comprobar disponibilidad del puerto  
   - Confirmar tokens de GitHub

2. Problemas con Azure Search  
   - Validar cadenas de conexión  
   - Verificar existencia del índice  
   - Confirmar carga de documentos

## Próximos pasos
- Explorar herramientas MCP adicionales  
- Implementar agentes personalizados  
- Mejorar capacidades RAG  
- Añadir más fuentes de eventos

## Recursos
- [MCP para principiantes](https://aka.ms/mcp-for-beginners)  
- [Documentación MCP](https://github.com/microsoft/semantic-kernel/tree/main/python/semantic-kernel/semantic_kernel/connectors/mcp)  
- [Documentación de Azure Cognitive Search](https://learn.microsoft.com/azure/search/)  
- [Guías de Semantic Kernel](https://learn.microsoft.com/semantic-kernel/)

**Aviso legal**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Aunque nos esforzamos por la precisión, tenga en cuenta que las traducciones automáticas pueden contener errores o inexactitudes. El documento original en su idioma nativo debe considerarse la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de ningún malentendido o interpretación errónea derivada del uso de esta traducción.