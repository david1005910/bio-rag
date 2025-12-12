"""RAG prompt templates for biomedical Q&A"""

# System prompt for biomedical research assistant
SYSTEM_PROMPT = """You are a biomedical research assistant specializing in analyzing scientific papers from PubMed.
Your role is to answer questions based ONLY on the provided research paper excerpts.

IMPORTANT RULES:
1. Only use information from the provided context - do not use prior knowledge
2. Always cite your sources using [PMID:xxxxx] format
3. If the context doesn't contain enough information, clearly state: "Based on the available papers, I cannot definitively answer this question"
4. Use accurate scientific terminology
5. Be concise but comprehensive
6. If you're uncertain about something, express that uncertainty
7. Provide at least 2 citations when possible

RESPONSE FORMAT:
- Start with a direct answer to the question
- Support your answer with evidence from the papers
- Include relevant citations inline using [PMID:xxxxx]
- End with a brief summary if the answer is long

LANGUAGE:
- Respond in the same language as the question
- For Korean questions, respond in Korean but keep scientific terms in English where appropriate
"""

# User prompt template
USER_TEMPLATE = """Based on the following research paper excerpts, please answer the question.

CONTEXT:
{context}

QUESTION: {question}

Please provide a comprehensive answer with proper citations:"""

# Context formatting template
CONTEXT_TEMPLATE = """[{index}] PMID: {pmid}
Title: {title}
Section: {section}
Content: {content}
"""

# Few-shot examples for better responses
FEW_SHOT_EXAMPLES = [
    {
        "question": "What are the main mechanisms of CAR-T cell therapy?",
        "context": """[1] PMID: 12345678
Title: Chimeric Antigen Receptor T-Cell Therapy: Mechanisms and Applications
Content: CAR-T cells are engineered to express chimeric antigen receptors that combine antigen recognition with T-cell activation domains. The CAR structure typically includes an extracellular antigen-binding domain derived from a single-chain variable fragment (scFv), a hinge region, a transmembrane domain, and intracellular signaling domains.""",
        "answer": """CAR-T cell therapy works through several key mechanisms:

1. **Antigen Recognition**: CAR-T cells are engineered with chimeric antigen receptors containing single-chain variable fragments (scFv) that recognize specific tumor antigens [PMID:12345678].

2. **T-cell Activation**: Upon antigen binding, the intracellular signaling domains (typically CD3ζ and costimulatory domains like CD28 or 4-1BB) trigger T-cell activation [PMID:12345678].

3. **Tumor Cell Killing**: Activated CAR-T cells directly kill tumor cells through cytotoxic mechanisms.

In summary, CAR-T therapy combines targeted antigen recognition with powerful T-cell effector functions to eliminate cancer cells.""",
    },
]

# Korean prompt template
SYSTEM_PROMPT_KO = """당신은 PubMed 과학 논문 분석을 전문으로 하는 바이오의학 연구 보조원입니다.
제공된 연구 논문 발췌문에만 기반하여 질문에 답변해야 합니다.

중요 규칙:
1. 제공된 컨텍스트의 정보만 사용하세요 - 사전 지식을 사용하지 마세요
2. 항상 [PMID:xxxxx] 형식으로 출처를 인용하세요
3. 컨텍스트에 충분한 정보가 없으면 명확히 밝히세요
4. 정확한 과학 용어를 사용하세요
5. 간결하지만 포괄적으로 답변하세요
6. 불확실한 부분은 그 불확실성을 표현하세요
7. 가능하면 최소 2개의 인용을 포함하세요
"""


def format_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into context string

    Args:
        chunks: List of chunk dictionaries with pmid, title, section, content

    Returns:
        Formatted context string
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            CONTEXT_TEMPLATE.format(
                index=i,
                pmid=chunk.get("pmid", "Unknown"),
                title=chunk.get("title", "Unknown"),
                section=chunk.get("section", "content"),
                content=chunk.get("content", ""),
            )
        )
    return "\n---\n".join(context_parts)


def build_prompt(question: str, context: str, language: str = "en") -> tuple[str, str]:
    """
    Build system and user prompts

    Args:
        question: User question
        context: Formatted context string
        language: Language code (en/ko)

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system = SYSTEM_PROMPT_KO if language == "ko" else SYSTEM_PROMPT
    user = USER_TEMPLATE.format(context=context, question=question)
    return system, user
