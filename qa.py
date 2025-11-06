
with tab5:
    st.header("💬 Q&A Assistant with Deep Dive")
    
    parsed = st.session_state.get("parsed")
    if not parsed:
        st.warning("Please upload and parse a repository first!")
    else:
        print(index_path)
        # if os.path.exists(index_path):
        try:
            idx, texts, meta, tokenized, vecs = load_index(index_path)
            st.success(f"🔍 Vector index loaded ({len(texts)} chunks)")
            
            # ✅ Initialize conversation memory in session state
            if "conversation_memory" not in st.session_state:
                from brd_generator import ConversationMemory
                st.session_state.conversation_memory = ConversationMemory(max_history=15)
            
            memory = st.session_state.conversation_memory
            
            # Show conversation stats
            if memory.history:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Conversation Turns", len(memory.history))
                with col2:
                    st.metric("Entities Discussed", len(memory.entities_discussed))
                with col3:
                    st.metric("Current Topic", memory.current_topic or "General")
            
            st.divider()
            
            # Q&A Interface with Deep Dive Toggle
            st.subheader("Ask Questions About Your Application")
            
            # Deep dive mode toggle
            col1, col2 = st.columns([3, 1])
            with col1:
                deep_dive_mode = st.checkbox(
                    "🔬 Deep Dive Mode",
                    value=False,
                    help="Get comprehensive, detailed answers with all available information"
                )
            with col2:
                if st.button("🔄 Clear Memory"):
                    st.session_state.conversation_memory = ConversationMemory(max_history=15)
                    st.success("Memory cleared!")
                    st.rerun()
            
            # Sample questions (categorized)
            st.write("**Sample Questions:**")
            
            with st.expander("💡 View Sample Questions by Category"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📊 Architecture & Structure**")
                    st.code("""
• What are ALL the main entities/tables?
• Show me the complete database schema
• What are all the business processes?
• What is the overall architecture?
                    """)
                    
                    st.markdown("**🔍 Deep Dive Questions**")
                    st.code("""
• Give me detailed information about [Entity]
• Show me all code related to [Process]
• What are the specific implementation details?
• List every dependency for [Component]
                    """)
                
                with col2:
                    st.markdown("**⚠️ Complexity & Risk**")
                    st.code("""
• Which files have the highest complexity?
• What are the technical debt areas?
• What will be hardest to migrate?
• What security concerns exist?
                    """)
                    
                    st.markdown("**⚡ Power Platform Mapping**")
                    st.code("""
• What Dataverse tables should be created?
• What Power Apps screens are needed?
• Show me all approval workflows
• What integrations exist?
                    """)
            
            # Show suggested follow-ups from previous answer
            if memory.history and 'last_follow_ups' in st.session_state:
                st.write("**💡 Suggested Follow-Up Questions:**")
                selected_follow_up = st.selectbox(
                    "Or click to use:",
                    [""] + st.session_state.last_follow_ups
                )
            else:
                selected_follow_up = ""
            
            # Question input
            question = st.text_area(
                "Your question:",
                value=selected_follow_up,
                placeholder="Ask anything... Use keywords like 'all', 'every', 'detail' for comprehensive answers.",
                height=100
            )
            
            col1, col2, col3 = st.columns([2, 2, 2])
            with col1:
                search_btn = st.button("🔍 Search & Answer", type="primary")
            with col2:
                show_context = st.checkbox("Show retrieved context", value=False)
            with col3:
                show_memory = st.checkbox("Show conversation context", value=False)
            
            if search_btn and question.strip():
                with st.spinner("Analyzing... (Deep dive mode may take longer)" if deep_dive_mode else "Searching and analyzing..."):
                    try:
                        from brd_generator import answer_question_with_memory
                        
                        # Call enhanced Q&A with memory
                        result = answer_question_with_memory(
                            question=question,
                            idx=idx,
                            texts=texts,
                            metadata=meta,
                            nodes=parsed["nodes"],
                            metrics=parsed.get("metrics"),
                            business_processes=parsed.get("business_processes"),
                            memory=memory,
                            deep_dive=deep_dive_mode
                        )
                        
                        # Show conversation context if requested
                        if show_memory and memory.history:
                            with st.expander("🧠 Conversation Context"):
                                st.text(memory.get_context_summary())
                        
                        # Show retrieved context if requested
                        if show_context:
                            st.subheader("📄 Retrieved Context")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Total Sources", result['total_context_docs'])
                            with col2:
                                st.metric("Confidence", f"{result['confidence']:.0%}")
                            
                            with st.expander(f"Top {len(result['sources'])} sources"):
                                for i, r in enumerate(result['sources']):
                                    doc_type = r.get('type', 'code')
                                    
                                    if doc_type == 'support_doc':
                                        title = r.get('file_name', 'N/A')
                                        st.markdown(f"**📄 Source {i+1}** - Document: {title} (Score: {r.get('score', 0):.3f})")
                                    else:
                                        path = r.get('path', r.get('file_name', '?'))
                                        st.markdown(f"**💻 Source {i+1}** - Code: {path} (Score: {r.get('score', 0):.3f})")
                                    
                                    text = r.get('text', '')
                                    if isinstance(text, dict):
                                        text = text.get('text', '')
                                    
                                    display_text = str(text)[:800]
                                    if len(str(text)) > 800:
                                        display_text += "..."
                                    
                                    st.code(display_text, language='csharp' if doc_type == 'code' else 'text')
                                    st.divider()
                        
                        # Main Answer
                        st.subheader("💡 Answer")
                        
                        # Show badges
                        badge_cols = st.columns(4)
                        with badge_cols[0]:
                            if result['is_follow_up']:
                                st.info("🔗 Follow-up Question")
                        with badge_cols[1]:
                            if deep_dive_mode:
                                st.info("🔬 Deep Dive")
                        with badge_cols[2]:
                            conf_color = "🟢" if result['confidence'] > 0.7 else "🟡" if result['confidence'] > 0.4 else "🔴"
                            st.info(f"{conf_color} Confidence: {result['confidence']:.0%}")
                        with badge_cols[3]:
                            st.info(f"📚 {result['total_context_docs']} sources")
                        
                        st.markdown(result['answer'])
                        
                        # Show suggested follow-ups
                        if result['suggested_follow_ups']:
                            st.divider()
                            st.subheader("💡 Suggested Follow-Up Questions")
                            
                            # Store in session state for next iteration
                            st.session_state.last_follow_ups = result['suggested_follow_ups']
                            
                            for follow_up in result['suggested_follow_ups']:
                                st.markdown(f"• {follow_up}")
                            
                            st.info("💡 Tip: Click one of these questions above to continue the conversation!")
                        
                        # Download button for answer
                        st.download_button(
                            "💾 Save Answer",
                            f"**Question:** {question}\n\n**Answer:**\n{result['answer']}\n\n**Sources:** {result['total_context_docs']}\n**Confidence:** {result['confidence']:.0%}",
                            f"qa_answer_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown"
                        )
                        
                    except Exception as e:
                        st.error(f"Q&A failed: {str(e)}")
                        st.exception(e)
            
            # Conversation History
            if memory.history:
                st.divider()
                st.subheader("📚 Conversation History")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    num_to_show = st.slider("Number of exchanges to show:", 1, min(10, len(memory.history)), 5)
                with col2:
                    if st.button("📥 Export History"):
                        history_text = ""
                        for i, qa in enumerate(memory.history, 1):
                            history_text += f"## Q{i}: {qa['question']}\n\n"
                            history_text += f"**Topic:** {qa['topic']}\n"
                            history_text += f"**Time:** {qa['timestamp']}\n\n"
                            history_text += f"**Answer:**\n{qa['answer']}\n\n"
                            history_text += "---\n\n"
                        
                        st.download_button(
                            "Download Conversation",
                            history_text,
                            f"conversation_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown"
                        )
                
                for i, qa in enumerate(reversed(memory.history[-num_to_show:]), 1):
                    with st.expander(f"Q{num_to_show - i + 1}: {qa['question'][:80]}... ({qa['topic']})"):
                        st.markdown(f"**🕐 {qa['timestamp'].strftime('%I:%M %p')}**")
                        st.markdown(f"**Question:** {qa['question']}")
                        st.markdown(f"**Answer:** {qa['answer'][:500]}..." if len(qa['answer']) > 500 else f"**Answer:** {qa['answer']}")
                        
                        if qa.get('retrieved_docs'):
                            st.caption(f"📊 Sources used: {len(qa['retrieved_docs'])}")
            
        except Exception as e:
            st.error(f"Failed to load vector index: {str(e)}")
            st.exception(e)











# brd_generator.py - ADD THESE ENHANCED Q&A FUNCTIONS

from typing import List, Dict, Any, Optional
import json
import re
import pandas as pd
# ============================================================
# âœ… CONVERSATIONAL MEMORY MANAGER
# ============================================================

class ConversationMemory:
    """Manages conversation history and context for deep-dive Q&A"""
    
    def __init__(self, max_history: int = 10):
        self.history: List[Dict[str, Any]] = []
        self.max_history = max_history
        self.current_topic = None
        self.entities_discussed = set()
        self.follow_up_count = 0
    
    def add_exchange(self, question: str, answer: str, retrieved_docs: List[Dict], topic: Optional[str] = None):
        """Add Q&A exchange to memory"""
        self.history.append({
            "question": question,
            "answer": answer,
            "retrieved_docs": retrieved_docs,
            "topic": topic or self._extract_topic(question),
            "timestamp": pd.Timestamp.now()
        })
        
        # Update entities discussed
        entities = self._extract_entities(question)
        self.entities_discussed.update(entities)
        
        # Trim history if needed
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def get_context_summary(self) -> str:
        """Generate summary of conversation so far"""
        if not self.history:
            return "No previous conversation."
        
        recent = self.history[-3:]  # Last 3 exchanges
        summary_parts = []
        
        for i, exchange in enumerate(recent, 1):
            summary_parts.append(
                f"Q{i}: {exchange['question'][:100]}...\n"
                f"A{i}: {exchange['answer'][:200]}..."
            )
        
        entities = ", ".join(list(self.entities_discussed)[-5:])
        
        return f"""
CONVERSATION CONTEXT:
Entities Discussed: {entities}
Current Topic: {self.current_topic or 'General'}

Recent Exchanges:
{chr(10).join(summary_parts)}
"""
    
    def _extract_topic(self, question: str) -> str:
        """Extract main topic from question"""
        q_lower = question.lower()
        
        topics = {
            'entity': ['entity', 'table', 'database', 'model'],
            'process': ['process', 'workflow', 'flow', 'approval'],
            'complexity': ['complex', 'difficult', 'risk', 'technical debt'],
            'screen': ['screen', 'view', 'form', 'ui', 'interface'],
            'security': ['security', 'authorization', 'role', 'permission']
        }
        
        for topic, keywords in topics.items():
            if any(kw in q_lower for kw in keywords):
                return topic
        
        return 'general'
    
    def _extract_entities(self, text: str) -> set:
        """Extract entity names from text"""
        # Look for capitalized words (likely entity names)
        entities = set(re.findall(r'\b[A-Z][a-zA-Z]+(?:[A-Z][a-zA-Z]+)*\b', text))
        return entities
    
    def is_follow_up(self, question: str) -> bool:
        """Detect if question is a follow-up"""
        if not self.history:
            return False
        
        follow_up_indicators = [
            'tell me more', 'what about', 'explain', 'elaborate',
            'give me details', 'deep dive', 'specific', 'how does',
            'why', 'show me', 'what are the', 'list'
        ]
        
        q_lower = question.lower()
        return any(indicator in q_lower for indicator in follow_up_indicators)
    
    def get_related_context(self, current_question: str) -> List[Dict]:
        """Get previously retrieved docs relevant to current question"""
        if not self.history:
            return []
        
        # Extract keywords from current question
        current_keywords = set(re.findall(r'\b\w+\b', current_question.lower()))
        
        related_docs = []
        for exchange in reversed(self.history[-5:]):  # Last 5 exchanges
            # Check if questions share keywords
            prev_keywords = set(re.findall(r'\b\w+\b', exchange['question'].lower()))
            overlap = current_keywords & prev_keywords
            
            if len(overlap) >= 2:  # At least 2 shared keywords
                related_docs.extend(exchange.get('retrieved_docs', []))
        
        # Deduplicate by hash
        seen_hashes = set()
        unique_docs = []
        for doc in related_docs:
            doc_hash = doc.get('hash', str(doc))
            if doc_hash not in seen_hashes:
                unique_docs.append(doc)
                seen_hashes.add(doc_hash)
        
        return unique_docs[:5]  # Top 5 related docs


# ============================================================
# âœ… DEEP DIVE RETRIEVAL STRATEGY
# ============================================================

def retrieve_with_expansion(
    idx, texts, meta,
    question: str,
    initial_top_k: int = 10,
    expand_if_needed: bool = True
) -> List[Dict[str, Any]]:
    """
    Retrieve with query expansion for deeper results.
    
    Strategy:
    1. Initial retrieval with original query
    2. If question asks for depth, expand query with synonyms
    3. Retrieve additional docs with expanded query
    4. Merge and deduplicate results
    """
    from rag_index import query as semantic_search
    
    # Initial retrieval
    results = semantic_search(idx, texts, meta, question, top_k=initial_top_k)
    
    # Detect if question needs expansion
    needs_expansion = any(indicator in question.lower() for indicator in [
        'all', 'every', 'each', 'detail', 'specific', 'deep dive',
        'comprehensive', 'complete', 'entire', 'full'
    ])
    
    if needs_expansion and expand_if_needed:
        print(f"Expanding query for deeper results...")
        
        # Generate expanded queries
        expanded_queries = _generate_expanded_queries(question)
        
        for exp_query in expanded_queries[:2]:  # Max 2 expansions
            exp_results = semantic_search(idx, texts, meta, exp_query, top_k=5)
            
            # Merge results (deduplicate by hash)
            existing_hashes = {r.get('hash', str(r)) for r in results}
            for exp_r in exp_results:
                exp_hash = exp_r.get('hash', str(exp_r))
                if exp_hash not in existing_hashes:
                    results.append(exp_r)
                    existing_hashes.add(exp_hash)
        
        print(f"âœ… Expanded to {len(results)} total results")
    
    return results


def _generate_expanded_queries(original_query: str) -> List[str]:
    """Generate expanded queries for deeper retrieval"""
    q_lower = original_query.lower()
    expanded = []
    
    # Synonym expansion
    expansions = {
        'entity': ['table', 'model', 'database table', 'data structure'],
        'table': ['entity', 'model', 'database', 'schema'],
        'process': ['workflow', 'business process', 'flow', 'operation'],
        'screen': ['view', 'form', 'page', 'interface', 'UI'],
        'complex': ['difficult', 'high complexity', 'risky', 'technical debt'],
        'approval': ['workflow', 'authorization', 'validation', 'review']
    }
    
    for term, synonyms in expansions.items():
        if term in q_lower:
            for syn in synonyms[:2]:  # Max 2 synonyms
                expanded.append(original_query.replace(term, syn))
    
    # Add specific detail queries
    if 'what' in q_lower:
        expanded.append(f"details about {original_query.replace('what', '').strip()}")
    
    if 'how' in q_lower:
        expanded.append(f"implementation of {original_query.replace('how', '').strip()}")
    
    return expanded[:3]  # Max 3 expanded queries


# ============================================================
# âœ… ENHANCED Q&A WITH MEMORY & DEEP DIVE
# ============================================================

def answer_question_with_memory(
    question: str,
    idx: Any,
    texts: List[str],
    metadata: List[Dict],
    nodes: List[Dict[str, Any]],
    metrics: Dict[str, Any] = None,
    business_processes: List[Dict[str, Any]] = None,
    memory: Optional[ConversationMemory] = None,
    deep_dive: bool = False
) -> Dict[str, Any]:
    """
    Enhanced Q&A with conversation memory and deep-dive capability.
    
    Args:
        question: User's question
        idx, texts, metadata: FAISS index components
        nodes, metrics, business_processes: Parsed data
        memory: ConversationMemory instance (optional)
        deep_dive: Force deep retrieval even for simple questions
    
    Returns:
        {
            "answer": str,
            "sources": List[Dict],
            "confidence": float,
            "is_follow_up": bool,
            "suggested_follow_ups": List[str]
        }
    """
    # Initialize memory if not provided
    if memory is None:
        memory = ConversationMemory()
    
    # Check if this is a follow-up question
    is_follow_up = memory.is_follow_up(question)
    
    # Determine retrieval strategy
    if deep_dive or is_follow_up:
        top_k = 15  # More docs for deep questions
        expand_query = True
    else:
        top_k = 8
        expand_query = False
    
    print(f"\nðŸ'¬ Processing question: {question[:80]}...")
    print(f"   Type: {'Follow-up' if is_follow_up else 'New'} | Deep dive: {deep_dive}")
    
    # Retrieve with expansion
    retrieved = retrieve_with_expansion(
        idx, texts, metadata,
        question,
        initial_top_k=top_k,
        expand_if_needed=expand_query
    )
    
    # Add related context from conversation history
    if is_follow_up:
        related_context = memory.get_related_context(question)
        print(f" Including {len(related_context)} docs from conversation history")
        
        # Merge (deduplicate)
        existing_hashes = {r.get('hash', str(r)) for r in retrieved}
        for ctx_doc in related_context:
            ctx_hash = ctx_doc.get('hash', str(ctx_doc))
            if ctx_hash not in existing_hashes:
                retrieved.append(ctx_doc)
    
    # Prepare comprehensive context
    graph_summary = summarize_graph_enhanced(nodes)
    context_snippets = _make_context_snippets(retrieved, max_chars=15000)
    
    conversation_context = ""
    if memory.history:
        conversation_context = memory.get_context_summary()
    
    additional_context = ""
    if metrics:
        additional_context += f"\n{format_metrics_summary(metrics)}"
    if business_processes:
        additional_context += f"\n{format_business_processes(business_processes)}"
    
    # Enhanced system prompt with depth instructions
    enhanced_system_prompt = f"""You are an expert code analyst helping migrate .NET applications to Power Platform.

RESPONSE STYLE:
{'- DEEP DIVE: Provide comprehensive, detailed analysis with specific examples.' if deep_dive else '- CONCISE: Provide clear, focused answers.'}
{'- FOLLOW-UP: Build upon previous answers, adding new insights.' if is_follow_up else '- STANDALONE: Provide complete context in your answer.'}

RULES:
1. Answer based ONLY on the provided context
2. If asking for "all" or "every", list ALL items found in context
3. Include specific file names, line numbers, or code snippets when relevant
4. If context is insufficient, say so clearly
5. For follow-up questions, reference previous discussion naturally

{conversation_context}
"""
    
    # Generate answer
    messages = [
        {"role": "system", "content": enhanced_system_prompt},
        {"role": "user", "content": f"""
QUESTION:
{question}

GRAPH STRUCTURE:
{graph_summary}

{additional_context}

RETRIEVED CODE CONTEXT:
{context_snippets}
"""}
    ]
    
    answer = _chat(messages, temperature=0.0)
    
    # Calculate confidence based on retrieval quality
    avg_score = sum(r.get('score', 0) for r in retrieved) / max(len(retrieved), 1)
    confidence = min(avg_score / 10, 1.0)  # Normalize to 0-1
    
    # Generate suggested follow-up questions
    follow_ups = _generate_follow_ups(question, answer, retrieved, memory)
    
    # Update memory
    topic = memory._extract_topic(question)
    memory.add_exchange(question, answer, retrieved, topic)
    
    return {
        "answer": answer,
        "sources": retrieved[:5],  # Top 5 sources
        "confidence": confidence,
        "is_follow_up": is_follow_up,
        "suggested_follow_ups": follow_ups,
        "total_context_docs": len(retrieved)
    }


def _generate_follow_ups(
    question: str,
    answer: str,
    retrieved: List[Dict],
    memory: ConversationMemory) -> List[str]:
    """Generate intelligent follow-up questions"""
    
    # Extract mentioned entities from answer
    entities = set(re.findall(r'\b[A-Z][a-zA-Z]+(?:[A-Z][a-zA-Z]+)*\b', answer))
    
    follow_ups = []
    
    # Entity-specific follow-ups
    if entities:
        entity = list(entities)[0]
        follow_ups.append(f"Show me all the code related to {entity}")
        follow_ups.append(f"What are the dependencies of {entity}?")
    
    # Topic-based follow-ups
    q_lower = question.lower()
    
    if 'complexity' in q_lower:
        follow_ups.append("Which specific files have the highest complexity?")
    elif 'process' in q_lower or 'workflow' in q_lower:
        follow_ups.append("Show me the detailed steps for this process")
    elif 'table' in q_lower or 'entity' in q_lower:
        follow_ups.append("What are all the columns and relationships?")
    elif 'screen' in q_lower or 'view' in q_lower:
        follow_ups.append("What fields and actions does this screen have?")
    
    # Generic deep-dive follow-ups
    if not follow_ups:
        follow_ups = [
            "Tell me more about the implementation details",
            "What are the potential migration challenges?",
            "Show me specific code examples"
        ]
    
    return follow_ups[:3]


# ============================================================
# âœ… BACKWARD COMPATIBILITY WRAPPER
# ============================================================

def answer_question_enhanced(
    question: str,
    retrieved: List[Dict[str, Any]],
    nodes: List[Dict[str, Any]],
    metrics: Dict[str, Any] = None,
    business_processes: List[Dict[str, Any]] = None) -> str:
    """
    Original function signature - now uses memory internally.
    For backward compatibility with existing code.
    """
    # Create temporary memory (not persisted across calls in this version)
    temp_memory = ConversationMemory(max_history=5)
    
    # Detect if this should be a deep dive
    deep_dive = any(indicator in question.lower() for indicator in [
        'all', 'every', 'detail', 'specific', 'deep dive',
        'comprehensive', 'list', 'show me'
    ])
    
    # Note: This simplified version doesn't persist memory across calls
    # For true memory, use answer_question_with_memory directly
    
    graph_summary = summarize_graph_enhanced(nodes)
    context = _make_context_snippets(retrieved, max_chars=15000)
    
    additional_context = ""
    if metrics:
        additional_context += f"\n{format_metrics_summary(metrics)}"
    if business_processes:
        additional_context += f"\n{format_business_processes(business_processes)}"
    
    system_prompt = QNA_SYSTEM_PROMPT
    if deep_dive:
        system_prompt += "\n\nIMPORTANT: This is a DEEP DIVE question. Provide comprehensive, detailed analysis with specific examples, file names, and code snippets."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"""
        QUESTION:
        {question}

        GRAPH:
        {graph_summary}

        {additional_context}

        CONTEXT:
        {context}
        """}
            ]
    
    return _chat(messages, temperature=0.0)
