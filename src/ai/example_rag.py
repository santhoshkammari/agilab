"""
Production RAG Pipeline Example
================================

Demonstrates how to use ai.py for a real-world RAG system with:
- Query classification
- Query refinement
- Search (mock)
- Answer generation
- Validation with chat history

Easy to modify when requirements change!
"""
import ai

# Configure once
lm = ai.LM(
    model="Qwen/Qwen3-4B-Instruct-2507",
    api_base="http://192.168.170.76:8000",
    temperature=0.1
)
ai.configure(lm)


class ProductionRAG(ai.Module):
    """Full RAG pipeline with all common steps."""

    def __init__(self):
        super().__init__()

        # Step 1: Query Classification
        self.classify = ai.Predict(
            "query -> classification",
            system="""You are an expert query classifier.
Classify queries into one of:
- SQL: For structured data queries (sales, metrics, counts)
- VECTOR: For semantic search (reviews, feedback, descriptions)

Return ONLY: SQL or VECTOR"""
        )

        # Step 2: Query Refinement
        self.refine = ai.Predict(
            "query, classification -> refined_query",
            system="""Refine the query for better retrieval.
- For SQL: Make it more specific with table/column hints
- For VECTOR: Extract key semantic concepts

Keep it concise."""
        )

        # Step 3: Answer Generation
        self.answer = ai.Predict(
            "context, query -> answer",
            system="""Answer the query based ONLY on the provided context.
Be concise and factual. If context is insufficient, say so."""
        )

        # Step 4: Validation (with history awareness)
        self.validate = ai.Predict(
            "query, answer, conversation_history -> validation",
            system="""Validate if the answer makes sense given:
1. The original query
2. The conversation history

Return:
- VALID: Answer is appropriate and consistent
- INVALID: Answer contradicts history or doesn't address query
- CONTEXT: Answer says "not enough context" appropriately

Return ONLY: VALID, INVALID, or CONTEXT"""
        )

    def forward(self, query: str, chat_history: list[dict] = None):
        """
        Main pipeline execution.

        Manager wants to change order? Just rearrange these lines!
        Manager wants to remove validation? Comment out Step 5!
        Manager wants to add a step? Insert it anywhere!
        """
        print(f"\n🔍 Processing: '{query}'")

        # Step 1: Classify
        classification = self.classify(query=query)
        print(f"  ├─ [Classify] {classification.strip()}")

        # Step 2: Refine
        refined = self.refine(
            query=query,
            classification=classification
        )
        print(f"  ├─ [Refine] {refined.strip()}")

        # Step 3: Search (mock - replace with real search)
        if "SQL" in classification.upper():
            context = self._sql_search(refined)
        else:
            context = self._vector_search(refined)
        print(f"  ├─ [Search] Retrieved {len(context)} chars")

        # Step 4: Generate answer
        answer = self.answer(
            context=context,
            query=refined
        )
        print(f"  ├─ [Answer] {answer.strip()[:80]}...")

        # Step 5: Validate with history (optional)
        if chat_history:
            history_text = "\n".join(
                f"{msg['role']}: {msg['content']}"
                for msg in chat_history
            )

            validation = self.validate(
                query=query,
                answer=answer,
                conversation_history=history_text,
                history=chat_history  # Also pass as conversation context
            )
            print(f"  └─ [Validate] {validation.strip()}")

            # Handle validation result
            if "INVALID" in validation.upper():
                print("  ⚠️  Validation failed! Regenerating...")
                return self._fallback_answer(query, chat_history)

        return answer.strip()

    def _sql_search(self, query: str) -> str:
        """Mock SQL search - replace with real DB query."""
        # Your SQL logic here
        return """
Sales Data:
- Q4 2023: $1.2M revenue, 450 orders
- Q1 2024: $1.4M revenue, 520 orders
- Top product: Widget Pro ($450K)
        """.strip()

    def _vector_search(self, query: str) -> str:
        """Mock vector search - replace with real embedding search."""
        # Your vector DB logic here
        return """
Customer Reviews:
- "Great product, very intuitive" (4.5/5)
- "Fast shipping, excellent support" (5/5)
- "Good value for money" (4/5)
        """.strip()

    def _fallback_answer(self, query: str, history: list[dict]) -> str:
        """Fallback when validation fails."""
        return "I'm not sure I understood correctly. Could you rephrase your question?"


# Example Usage
if __name__ == "__main__":
    pipeline = ProductionRAG()

    # Scenario 1: Simple query
    print("\n" + "="*60)
    print("SCENARIO 1: Simple Query")
    print("="*60)
    answer = pipeline(query="What were our Q4 sales?")
    print(f"\n✅ Final Answer: {answer}\n")

    # Scenario 2: With conversation history
    print("="*60)
    print("SCENARIO 2: With Chat History")
    print("="*60)
    history = [
        {"role": "user", "content": "I'm analyzing 2023 performance"},
        {"role": "assistant", "content": "I can help with 2023 data. What would you like to know?"}
    ]
    answer = pipeline(
        query="How did we do in the last quarter?",
        chat_history=history
    )
    print(f"\n✅ Final Answer: {answer}\n")

    # Scenario 3: Vector search
    print("="*60)
    print("SCENARIO 3: Semantic Query")
    print("="*60)
    answer = pipeline(query="What do customers think about us?")
    print(f"\n✅ Final Answer: {answer}\n")

    # Inspect what happened
    print("="*60)
    print("PIPELINE INSPECTION")
    print("="*60)
    print("\nClassifier history:")
    pipeline.inspect_history("classify")
