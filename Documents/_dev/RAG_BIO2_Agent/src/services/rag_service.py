import os
import re
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from openai import OpenAI

from .embedding_service import EmbeddingService
from .vector_store import QdrantVectorStore, SearchResult

@dataclass
class RAGResponse:
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    chunks_used: List[SearchResult]
    reasoning_steps: List[Dict[str, Any]] = None

SYSTEM_PROMPT = """You are an expert biomedical researcher assistant with deep knowledge in molecular biology, genetics, pharmacology, and medical research. Your role is to help researchers understand complex scientific papers and provide accurate, evidence-based answers.

IMPORTANT RULES:
1. Only use information from the provided research paper context
2. Always cite sources using [PMID: xxxxx] format when referencing specific papers
3. If the context doesn't contain enough information, clearly state "I cannot find sufficient information in the provided papers to fully answer this question"
4. Do not make assumptions or add information not present in the context
5. Explain complex biomedical terms when they first appear
6. Be precise and factual in your responses
7. Acknowledge limitations in the available data when appropriate

Format your response clearly with proper paragraphs and use bullet points for lists when appropriate."""

class RAGService:
    def __init__(
        self,
        embedding_model: str = "pubmedbert",
        vector_dimension: int = 768,
        collection_name: str = None
    ):
        self.embedding_service = EmbeddingService(model_type=embedding_model)
        self.vector_store = QdrantVectorStore(
            collection_name=collection_name,
            dimension=vector_dimension
        )
        self._translator = None
        self._init_llm_client()
    
    def _get_translator(self):
        if self._translator is None:
            try:
                from .translation_service import TranslationService
                self._translator = TranslationService()
            except Exception:
                pass
        return self._translator
    
    def _translate_if_korean(self, text: str) -> str:
        translator = self._get_translator()
        if translator and translator.is_korean(text):
            return translator.translate_to_english(text)
        return text
    
    def _init_llm_client(self):
        """Initialize LLM client. LLM is optional - only needed for query, not indexing."""
        self.llm_client = None
        base_url = os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL")
        api_key = os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY")

        if base_url and api_key:
            self.llm_client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
        else:
            openai_key = os.environ.get("OPENAI_API_KEY")
            if openai_key:
                self.llm_client = OpenAI(api_key=openai_key)
            else:
                print("Warning: No LLM API key found. Query functionality will be limited.")

    def _require_llm(self):
        """Check if LLM client is available, raise error if not."""
        if not self.llm_client:
            raise ValueError("No LLM API key found. Please configure OpenAI integration for query functionality.")
    
    def index_paper(
        self,
        pmid: str,
        title: str,
        abstract: str,
        authors: List[str] = None,
        journal: str = None,
        publication_date: str = None,
        full_text: str = None
    ) -> List[str]:
        chunks = self._create_chunks(pmid, title, abstract, full_text)
        
        texts = [c["text"] for c in chunks]
        embeddings = self.embedding_service.batch_encode(texts)
        
        metadatas = []
        for chunk in chunks:
            metadatas.append({
                "pmid": pmid,
                "title": title,
                "section": chunk["section"],
                "authors": ", ".join(authors) if authors else "",
                "journal": journal or "",
                "publication_date": publication_date or ""
            })
        
        ids = self.vector_store.add_documents(texts, embeddings, metadatas)
        return ids
    
    def _create_chunks(
        self,
        pmid: str,
        title: str,
        abstract: str,
        full_text: str = None,
        max_chunk_size: int = 500
    ) -> List[Dict[str, str]]:
        chunks = []
        
        if title:
            chunks.append({
                "text": f"Title: {title}",
                "section": "title"
            })
        
        if abstract:
            if len(abstract) > max_chunk_size * 2:
                sentences = re.split(r'(?<=[.!?])\s+', abstract)
                current_chunk = ""
                chunk_idx = 0
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) < max_chunk_size:
                        current_chunk += " " + sentence
                    else:
                        if current_chunk.strip():
                            chunks.append({
                                "text": current_chunk.strip(),
                                "section": f"abstract_{chunk_idx}"
                            })
                            chunk_idx += 1
                        current_chunk = sentence
                
                if current_chunk.strip():
                    chunks.append({
                        "text": current_chunk.strip(),
                        "section": f"abstract_{chunk_idx}"
                    })
            else:
                chunks.append({
                    "text": abstract,
                    "section": "abstract"
                })
        
        if full_text:
            sections = self._extract_sections(full_text)
            for section_name, section_text in sections.items():
                if len(section_text) > max_chunk_size:
                    sub_chunks = self._split_text(section_text, max_chunk_size)
                    for i, sub_chunk in enumerate(sub_chunks):
                        chunks.append({
                            "text": sub_chunk,
                            "section": f"{section_name}_{i}"
                        })
                else:
                    chunks.append({
                        "text": section_text,
                        "section": section_name
                    })
        
        return chunks
    
    def _extract_sections(self, text: str) -> Dict[str, str]:
        section_patterns = {
            'introduction': r'(?i)(?:INTRODUCTION|1\.\s*Introduction)',
            'methods': r'(?i)(?:METHODS|MATERIALS?\s*AND\s*METHODS|2\.\s*Methods)',
            'results': r'(?i)(?:RESULTS|3\.\s*Results)',
            'discussion': r'(?i)(?:DISCUSSION|4\.\s*Discussion)',
            'conclusion': r'(?i)(?:CONCLUSION|CONCLUSIONS|5\.\s*Conclusion)'
        }
        
        return {"full_text": text}
    
    def _split_text(self, text: str, max_size: int) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_size:
                current_chunk += " " + sentence
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    async def query(
        self,
        question: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> RAGResponse:
        search_query = self._translate_if_korean(question)
        question_embedding = self.embedding_service.encode(search_query)

        search_results = self.vector_store.search(
            query_embedding=question_embedding,
            top_k=top_k,
            filter_dict=filter_dict
        )

        if not search_results:
            return RAGResponse(
                answer="I couldn't find any relevant papers in the database for your question. Please try a different query or add more papers to the database.",
                sources=[],
                confidence=0.0,
                chunks_used=[]
            )

        context = self._build_context(search_results)
        sources = self._extract_sources(search_results)

        # Try to generate answer with LLM, fallback to context summary if no LLM
        if self.llm_client:
            answer = await self._generate_answer(question, context)
            confidence = self._calculate_confidence(search_results, answer)
        else:
            # No LLM available - provide search results as answer
            answer = self._generate_fallback_answer(question, search_results)
            confidence = sum(r.score for r in search_results) / len(search_results) if search_results else 0.0

        return RAGResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
            chunks_used=search_results
        )

    def _generate_fallback_answer(self, question: str, results: List[SearchResult]) -> str:
        """Generate a simple answer from search results when LLM is not available."""
        if not results:
            return "No relevant information found."

        answer_parts = [f"Based on {len(results)} relevant papers found for your question:\n"]

        for i, result in enumerate(results[:3], 1):
            pmid = result.metadata.get("pmid", "Unknown")
            title = result.metadata.get("title", "Unknown")
            text = result.text[:300] + "..." if len(result.text) > 300 else result.text
            answer_parts.append(f"\n**[{i}] {title}** (PMID: {pmid})\n{text}\n")

        answer_parts.append("\n*Note: For AI-generated answers, please configure an OpenAI API key.*")
        return "".join(answer_parts)
    
    def _build_context(self, results: List[SearchResult]) -> str:
        context_parts = []
        
        for i, result in enumerate(results, 1):
            pmid = result.metadata.get("pmid", "Unknown")
            title = result.metadata.get("title", "Unknown")
            section = result.metadata.get("section", "Unknown")
            
            context_parts.append(
                f"[Paper {i}] PMID: {pmid}\n"
                f"Title: {title}\n"
                f"Section: {section}\n"
                f"Content: {result.text}\n"
            )
        
        return "\n\n".join(context_parts)
    
    async def _generate_answer(self, question: str, context: str) -> str:
        self._require_llm()
        user_prompt = f"""Based on the following research paper excerpts, please answer the question.

Context from research papers:
{context}

Question: {question}

Please provide a detailed, accurate answer with citations to the relevant papers using [PMID: xxxxx] format:"""

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2000,
                temperature=0.1
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _extract_sources(self, results: List[SearchResult]) -> List[Dict[str, Any]]:
        sources = []
        seen_pmids = set()
        
        for result in results:
            pmid = result.metadata.get("pmid")
            if pmid and pmid not in seen_pmids:
                seen_pmids.add(pmid)
                sources.append({
                    "pmid": pmid,
                    "title": result.metadata.get("title", ""),
                    "journal": result.metadata.get("journal", ""),
                    "relevance": result.score,
                    "excerpt": result.text[:200] + "..." if len(result.text) > 200 else result.text
                })
        
        return sources
    
    def _calculate_confidence(self, results: List[SearchResult], answer: str) -> float:
        if not results:
            return 0.0
        
        avg_score = sum(r.score for r in results) / len(results)
        
        cited_pmids = re.findall(r'PMID:\s*(\d+)', answer)
        source_pmids = [r.metadata.get("pmid") for r in results]
        
        if cited_pmids:
            valid_citations = sum(1 for pmid in cited_pmids if pmid in source_pmids)
            citation_score = valid_citations / len(cited_pmids)
        else:
            citation_score = 0.5
        
        confidence = (avg_score * 0.6) + (citation_score * 0.4)
        return min(confidence, 1.0)
    
    def semantic_search(
        self,
        query: str,
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        search_query = self._translate_if_korean(query)
        query_embedding = self.embedding_service.encode(search_query)
        
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k * 2,
            filter_dict=filter_dict
        )
        
        papers = {}
        for result in results:
            pmid = result.metadata.get("pmid")
            if pmid not in papers:
                papers[pmid] = {
                    "pmid": pmid,
                    "title": result.metadata.get("title", ""),
                    "journal": result.metadata.get("journal", ""),
                    "publication_date": result.metadata.get("publication_date", ""),
                    "relevance": result.score,
                    "excerpt": result.text[:300] + "..." if len(result.text) > 300 else result.text
                }
            else:
                papers[pmid]["relevance"] = max(papers[pmid]["relevance"], result.score)
        
        return sorted(papers.values(), key=lambda x: x["relevance"], reverse=True)[:top_k]
    
    async def reasoning_query(
        self,
        question: str,
        top_k: int = 5,
        max_iterations: int = 3,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> RAGResponse:
        reasoning_steps = []
        all_search_results = []
        accumulated_context = []
        
        step1 = await self._decompose_question(question)
        reasoning_steps.append({
            "step": 1,
            "type": "decomposition",
            "description": "질문 분해 및 분석",
            "content": step1
        })
        
        sub_questions = step1.get("sub_questions", [question])
        
        for i, sub_q in enumerate(sub_questions[:max_iterations]):
            search_sub_q = self._translate_if_korean(sub_q)
            query_embedding = self.embedding_service.encode(search_sub_q)
            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                filter_dict=filter_dict
            )
            
            for r in results:
                if r not in all_search_results:
                    all_search_results.append(r)
            
            if results:
                context = self._build_context(results)
                accumulated_context.append(f"[Sub-question {i+1}]: {sub_q}\n\n{context}")
                
                sub_answer = await self._generate_sub_answer(sub_q, context)
                reasoning_steps.append({
                    "step": i + 2,
                    "type": "sub_answer",
                    "description": f"하위 질문 {i+1} 분석",
                    "sub_question": sub_q,
                    "content": sub_answer,
                    "sources_found": len(results)
                })
        
        if not all_search_results:
            return RAGResponse(
                answer="인덱싱된 논문에서 관련 정보를 찾을 수 없습니다. 다른 질문을 시도하거나 더 많은 논문을 인덱싱해 주세요.",
                sources=[],
                confidence=0.0,
                chunks_used=[],
                reasoning_steps=reasoning_steps
            )
        
        full_context = "\n\n---\n\n".join(accumulated_context)
        final_answer = await self._synthesize_reasoning_answer(question, full_context, reasoning_steps)
        
        reasoning_steps.append({
            "step": len(reasoning_steps) + 1,
            "type": "synthesis",
            "description": "최종 답변 종합",
            "content": "모든 하위 분석 결과를 종합하여 최종 답변 생성"
        })
        
        sources = self._extract_sources(all_search_results)
        confidence = self._calculate_confidence(all_search_results, final_answer)
        confidence = min(confidence * 1.1, 1.0)
        
        return RAGResponse(
            answer=final_answer,
            sources=sources,
            confidence=confidence,
            chunks_used=all_search_results,
            reasoning_steps=reasoning_steps
        )
    
    async def _decompose_question(self, question: str) -> Dict[str, Any]:
        self._require_llm()
        decompose_prompt = f"""You are an expert at analyzing complex biomedical research questions.
Given the following question, analyze it and break it down into simpler sub-questions that can be answered individually.

Original Question: {question}

Please respond in JSON format with:
{{
    "complexity": "simple" | "moderate" | "complex",
    "main_concepts": ["concept1", "concept2", ...],
    "sub_questions": ["sub-question 1", "sub-question 2", ...],
    "reasoning_approach": "brief description of how to approach this question"
}}

If the question is simple, just return the original question as the only sub-question.
Return ONLY valid JSON, no other text."""

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": decompose_prompt}
                ],
                max_tokens=1000,
                temperature=0.2
            )
            
            result_text = response.choices[0].message.content.strip()
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
            
            return json.loads(result_text)
        except Exception as e:
            return {
                "complexity": "simple",
                "main_concepts": [],
                "sub_questions": [question],
                "reasoning_approach": "Direct search and answer"
            }
    
    async def _generate_sub_answer(self, question: str, context: str) -> str:
        self._require_llm()
        prompt = f"""Based on the following research paper excerpts, briefly answer the question.
Focus on extracting key facts and findings relevant to the question.

Context:
{context}

Question: {question}

Provide a concise, factual answer with citations [PMID: xxxxx]:"""

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a biomedical research assistant. Provide concise, evidence-based answers."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def _synthesize_reasoning_answer(
        self,
        original_question: str,
        accumulated_context: str,
        reasoning_steps: List[Dict[str, Any]]
    ) -> str:
        self._require_llm()
        steps_summary = "\n".join([
            f"- Step {s['step']}: {s['description']}" + 
            (f"\n  Finding: {s.get('content', '')[:200]}..." if s.get('content') else "")
            for s in reasoning_steps if s['type'] == 'sub_answer'
        ])
        
        synthesis_prompt = f"""You are an expert biomedical researcher. Based on the multi-step analysis below, provide a comprehensive answer to the original question.

ORIGINAL QUESTION: {original_question}

ANALYSIS STEPS AND FINDINGS:
{steps_summary}

FULL CONTEXT FROM RESEARCH PAPERS:
{accumulated_context}

INSTRUCTIONS:
1. Synthesize all findings into a coherent, comprehensive answer
2. Highlight key insights and connections between different pieces of evidence
3. Cite all relevant sources using [PMID: xxxxx] format
4. If there are conflicting findings, acknowledge and discuss them
5. Provide a clear conclusion

Please provide your synthesized answer:"""

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT + "\n\nYou are performing multi-step reasoning to answer complex questions. Synthesize evidence from multiple searches to provide a comprehensive answer."},
                    {"role": "user", "content": synthesis_prompt}
                ],
                max_tokens=2500,
                temperature=0.15
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error synthesizing answer: {str(e)}"
