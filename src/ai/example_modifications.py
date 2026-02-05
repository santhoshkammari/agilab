"""
Manager Change Scenarios - How Easy It Is!
===========================================

Your manager constantly changes requirements?
This shows how simple it is to modify ai.py pipelines.
"""
import ai

# Setup
lm = ai.LM(
    model="Qwen/Qwen3-4B-Instruct-2507",
    api_base="http://192.168.170.76:8000",
    temperature=0.1
)
ai.configure(lm)


# ============================================================
# SCENARIO 1: Manager says "Remove the refine step"
# ============================================================
print("\n" + "="*60)
print("SCENARIO 1: Original Pipeline")
print("="*60)

class OriginalRAG(ai.Module):
    def __init__(self):
        super().__init__()
        self.classify = ai.Predict("query -> classification", system="Classify as SQL or VECTOR")
        self.refine = ai.Predict("query -> refined_query", system="Refine the query")
        self.answer = ai.Predict("context, query -> answer")

    def forward(self, query):
        print(f"Processing: {query}")
        c = self.classify(query=query)
        print(f"  ├─ Classified: {c.strip()}")

        r = self.refine(query=query)
        print(f"  ├─ Refined: {r.strip()}")

        context = "Mock context"
        ans = self.answer(context=context, query=r)
        print(f"  └─ Answer: {ans.strip()[:50]}...")
        return ans

pipeline = OriginalRAG()
pipeline("What are our sales?")


print("\n" + "="*60)
print("SCENARIO 1: After Manager's Request (2 minutes later)")
print("="*60)

class SimplifiedRAG(ai.Module):
    def __init__(self):
        super().__init__()
        self.classify = ai.Predict("query -> classification", system="Classify as SQL or VECTOR")
        # self.refine = ai.Predict("query -> refined_query", system="Refine the query")  # COMMENTED OUT!
        self.answer = ai.Predict("context, query -> answer")

    def forward(self, query):
        print(f"Processing: {query}")
        c = self.classify(query=query)
        print(f"  ├─ Classified: {c.strip()}")

        # r = self.refine(query=query)  # COMMENTED OUT!
        # print(f"  ├─ Refined: {r.strip()}")

        context = "Mock context"
        ans = self.answer(context=context, query=query)  # Use original query
        print(f"  └─ Answer: {ans.strip()[:50]}...")
        return ans

pipeline2 = SimplifiedRAG()
pipeline2("What are our sales?")


# ============================================================
# SCENARIO 2: Manager says "Move validation before answer"
# ============================================================
print("\n\n" + "="*60)
print("SCENARIO 2: Reorder Steps")
print("="*60)

class ReorderedRAG(ai.Module):
    def __init__(self):
        super().__init__()
        self.classify = ai.Predict("query -> classification")
        self.validate_input = ai.Predict("query -> validation", system="Check if query is clear")
        self.answer = ai.Predict("query -> answer")

    def forward(self, query):
        print(f"Processing: {query}")

        # Step 1: Classify
        c = self.classify(query=query)
        print(f"  ├─ Classified: {c.strip()[:30]}...")

        # Step 2: Validate BEFORE answering (moved up!)
        v = self.validate_input(query=query)
        print(f"  ├─ Validation: {v.strip()[:30]}...")

        if "invalid" in v.lower():
            return "Please clarify your question"

        # Step 3: Answer (only if valid)
        ans = self.answer(query=query)
        print(f"  └─ Answer: {ans.strip()[:50]}...")
        return ans

pipeline3 = ReorderedRAG()
pipeline3("What's the thing?")


# ============================================================
# SCENARIO 3: Manager says "Change the prompt for classifier"
# ============================================================
print("\n\n" + "="*60)
print("SCENARIO 3: Prompt Engineering (Every 2 Hours)")
print("="*60)

print("\nVersion 1: Simple prompt")
class V1(ai.Module):
    def __init__(self):
        super().__init__()
        self.classify = ai.Predict(
            "query -> classification",
            system="Classify as SQL or VECTOR"
        )

    def forward(self, query):
        return self.classify(query=query)

v1 = V1()
result = v1("Show customer feedback")
print(f"Result: {result.strip()}")


print("\nVersion 2: Detailed prompt (manager changed mind)")
class V2(ai.Module):
    def __init__(self):
        super().__init__()
        self.classify = ai.Predict(
            "query -> classification",
            system="""Expert classifier. Rules:
- SQL: numeric queries, aggregations, filters
- VECTOR: opinions, descriptions, semantic search
Think carefully. Return ONLY: SQL or VECTOR"""
        )

    def forward(self, query):
        return self.classify(query=query)

v2 = V2()
result = v2("Show customer feedback")
print(f"Result: {result.strip()}")


# ============================================================
# SCENARIO 4: Manager says "Add logging everywhere"
# ============================================================
print("\n\n" + "="*60)
print("SCENARIO 4: Add Logging (Already Built In!)")
print("="*60)

class LoggedRAG(ai.Module):
    def __init__(self):
        super().__init__()
        self.classify = ai.Predict("query -> classification")
        self.answer = ai.Predict("query -> answer")

    def forward(self, query):
        # Just print at each step - that's it!
        print(f"\n[LOG] Input: {query}")

        c = self.classify(query=query)
        print(f"[LOG] Classification: {c.strip()}")

        ans = self.answer(query=query)
        print(f"[LOG] Answer: {ans.strip()}")

        return ans

pipeline4 = LoggedRAG()
pipeline4("What's the weather?")


# ============================================================
# SCENARIO 5: Compare different approaches
# ============================================================
print("\n\n" + "="*60)
print("SCENARIO 5: A/B Testing Different Prompts")
print("="*60)

dataset = [
    {'input': 'Show me sales numbers', 'expected': 'SQL'},
    {'input': 'What do users say?', 'expected': 'VECTOR'}
]

def check_classification(example, prediction):
    return example['expected'].lower() in prediction.lower()

# Approach A: Simple
classifier_a = ai.Predict("query -> classification", system="Classify: SQL or VECTOR")
eval_a = ai.Eval(check_classification, dataset, save_path="/tmp/approach_a.json")
result_a = eval_a(classifier_a)
print(f"Approach A: {result_a['score']}%")

# Approach B: Detailed
classifier_b = ai.Predict(
    "query -> classification",
    system="Expert SQL/VECTOR classifier. SQL=numbers, VECTOR=text."
)
eval_b = ai.Eval(check_classification, dataset, save_path="/tmp/approach_b.json")
result_b = eval_b(classifier_b)
print(f"Approach B: {result_b['score']}%")
print(f"Improvement: {result_b['score'] - result_a['score']}%")


print("\n\n" + "="*60)
print("SUMMARY: Changes That Take < 5 Minutes")
print("="*60)
print("""
✅ Remove a step: Comment out 2-3 lines
✅ Reorder steps: Cut & paste in forward()
✅ Change prompt: Edit system= parameter
✅ Add logging: Add print() statements
✅ Add validation: Insert new Predict + if statement
✅ A/B test: Create 2 Predict objects, compare with Eval
✅ Change model: Just change LM config once

This is why ai.py is production-ready for fast-moving teams!
""")
