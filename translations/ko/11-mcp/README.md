<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9320dd53c82869fd44935d1581eaf7bb",
  "translation_date": "2025-05-21T08:12:03+00:00",
  "source_file": "11-mcp/README.md",
  "language_code": "ko"
}
-->
# Lesson 11: Model Context Protocol (MCP) Integration

## Learning Objectives
- MCP가 무엇인지와 AI 에이전트 개발에서의 역할 이해하기
- GitHub 통합을 위한 MCP 서버 설정 및 구성하기
- MCP 도구를 사용한 다중 에이전트 시스템 구축하기
- Azure Cognitive Search를 활용한 RAG(검색 기반 생성) 구현하기

## Prerequisites
- Python 3.8 이상
- Node.js 14 이상
- Azure 구독
- GitHub 계정
- Semantic Kernel에 대한 기본 이해

## Setup Instructions

1. **환경 설정**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Azure 서비스 구성**
   - Azure Cognitive Search 리소스 생성
   - Azure OpenAI 서비스 설정
   - `.env` 파일에 환경 변수 구성

3. **MCP 서버 설정**
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

### 1. 다중 에이전트 시스템
- GitHub Agent: 저장소 분석
- Hackathon Agent: 프로젝트 추천
- Events Agent: 기술 행사 추천

### 2. Azure 통합
- 이벤트 인덱싱을 위한 Cognitive Search
- 에이전트 인텔리전스를 위한 Azure OpenAI
- RAG 패턴 구현

### 3. MCP 도구
- GitHub 저장소 분석
- 코드 검사
- 메타데이터 추출

## Code Walkthrough

샘플에서는 다음을 보여줍니다:
1. MCP 서버 통합
2. 다중 에이전트 오케스트레이션
3. Azure Cognitive Search 통합
4. RAG 패턴 구현

주요 기능:
- 실시간 GitHub 저장소 분석
- 지능형 프로젝트 추천
- Azure Search를 활용한 이벤트 매칭
- Chainlit을 통한 스트리밍 응답

## Running the Sample

1. MCP 서버 시작:
   ```bash
   npx @modelcontextprotocol/server-github
   ```

2. 애플리케이션 실행:
   ```bash
   chainlit run app.py -w
   ```

3. 통합 테스트:
   ```
   Example query: "Analyze repositories for username: <github_username>"
   ```

## Troubleshooting

자주 발생하는 문제와 해결 방법:
1. MCP 연결 문제
   - 서버가 실행 중인지 확인
   - 포트 사용 가능 여부 점검
   - GitHub 토큰 확인

2. Azure Search 문제
   - 연결 문자열 유효성 검사
   - 인덱스 존재 여부 확인
   - 문서 업로드 확인

## Next Steps
- 추가 MCP 도구 탐색
- 맞춤형 에이전트 구현
- RAG 기능 강화
- 더 많은 이벤트 소스 추가

## Resources
- [MCP for Beginners](https://aka.ms/mcp-for-beginners)  
- [MCP Documentation](https://github.com/microsoft/semantic-kernel/tree/main/python/semantic-kernel/semantic_kernel/connectors/mcp)
- [Azure Cognitive Search Docs](https://learn.microsoft.com/azure/search/)
- [Semantic Kernel Guides](https://learn.microsoft.com/semantic-kernel/)

**면책 조항**:  
이 문서는 AI 번역 서비스 [Co-op Translator](https://github.com/Azure/co-op-translator)를 사용하여 번역되었습니다. 정확성을 위해 최선을 다하고 있으나, 자동 번역에는 오류나 부정확성이 포함될 수 있음을 유의해 주시기 바랍니다. 원본 문서의 원어 버전이 권위 있는 자료로 간주되어야 합니다. 중요한 정보의 경우 전문적인 인간 번역을 권장합니다. 본 번역 사용으로 인한 오해나 잘못된 해석에 대해 당사는 책임을 지지 않습니다.