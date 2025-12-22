# Text-to-SQL with JSON AST + GRPO

This document summarizes the **findings, design decisions, architecture, and code** for using a **JSON-based AST (Intermediate Representation)** for Text-to-SQL, optimized for **GRPO / RL-style fine-tuning**.

---

## 1. Motivation

Directly training models to emit raw SQL strings is fragile:

- SQL is **non-canonical** (many strings, same semantics)
- Token-level rewards are noisy
- Formatting differences dominate learning signal
- GRPO becomes unstable

**Key insight**:  
> SQL should be a *compiled artifact*, not the learning target.

Instead, train on a **structured semantic representation** → compile to SQL afterward.

---

## 2. Core Idea

Use a **JSON Abstract Syntax Tree (AST)** as the model output.

Model learns:
- relations
- projections
- filters
- operators
- composition

Compiler handles:
- SQL syntax
- formatting
- vendor quirks

This dramatically reduces reward variance.

---

## 3. Why JSON AST (and not raw SQL or XML)

### Compared to SQL text
- Deterministic structure
- Semantic comparison possible
- Easier partial credit

### Compared to XML
- Less verbose
- Better tokenizer alignment
- Native support in tool-calling / structured output

**Conclusion**: JSON AST is the sweet spot.

---

## 4. Design Principles for the AST

1. **Semantic, not syntactic**
2. **Composable / recursive**
3. **Minimal but extensible**
4. **Stable under roundtrip (AST → SQL → AST)**

Do NOT mirror SQL grammar directly.

---

## 5. Minimal Supported SQL Subset (v1)

This prototype supports:

- SELECT
- multiple columns
- FROM
- WHERE with simple operators (=, >, <)
- string & integer literals

This already covers the majority of benchmark queries.

---

## 6. Reference JSON AST Schema (v1)

```json
{
  "select": ["column1", "column2"],
  "from": "table_name",
  "where": {
    "left": "column",
    "op": ">",
    "right": 10
  }
}
```

Notes:
- `where` is optional
- Operators are explicit
- Literals are typed

---

## 7. Python Reference Implementation

### jsonast_to_sql(ast)

```python
def jsonast_to_sql(ast):
    select_clause = ", ".join(ast["select"])
    from_clause = ast["from"]
    sql = f"SELECT {select_clause} FROM {from_clause}"
    if ast.get("where"):
        w = ast["where"]
        sql += f" WHERE {w['left']} {w['op']} {repr(w['right'])}"
    return sql + ";"
```

---

### sql_to_jsonast(sql)

```python
import re

def sql_to_jsonast(sql):
    sql = sql.strip().rstrip(";")
    pattern = r"SELECT (.+) FROM (\w+)(?: WHERE (\w+)\s*(=|>|<)\s*('?[^']+'?|\d+))?"
    m = re.match(pattern, sql, re.IGNORECASE)
    if not m:
        raise ValueError("Unsupported SQL")

    select, table, left, op, right = m.groups()
    ast = {
        "select": [c.strip() for c in select.split(",")],
        "from": table
    }

    if left:
        try:
            right_val = int(right.strip("'"))
        except:
            right_val = right.strip("'")
        ast["where"] = {
            "left": left,
            "op": op,
            "right": right_val
        }

    return ast
```

---

## 8. Roundtrip Testing Strategy

### SQL → AST → SQL
- ensures compiler determinism
- catches ambiguity early

### AST → SQL → AST
- validates parser stability
- guarantees reward consistency

Roundtrip stability is **critical** for GRPO.

---

## 9. GRPO Training Loop (Conceptual)

1. Model outputs JSON AST
2. Validator checks:
   - JSON schema validity
   - column/table existence
3. Compiler converts AST → SQL
4. Optional execution against DB
5. Reward shaping:
   - +1 exact AST match
   - +0.5 semantic match (same result)
   - −1 invalid structure

**Never reward raw SQL token overlap.**

---

## 10. Escape Hatch (Advanced)

For rare unsupported queries:

```json
{
  "raw_sql": "SELECT ..."
}
```

- Penalize lightly
- Log for schema expansion
- Do NOT block training

This preserves stability while allowing coverage growth.

---

## 11. What NOT to Do

- ❌ Train directly on SQL strings
- ❌ Optimize formatting
- ❌ Chase full SQL coverage early
- ❌ Encode vendor-specific syntax

---

## 12. Roadmap

Safe expansion order:

1. AND / OR filters
2. JOINs
3. GROUP BY + HAVING
4. Subqueries (recursive AST)
5. Execution-based rewards

Each step:
- update schema
- add roundtrip tests
- retrain with mixed old + new data

---

## 13. Final Takeaway

- JSON AST **can cover all practical Text-to-SQL**
- Structure is the real supervision signal
- SQL is just a rendering
- GRPO becomes stable, low-variance, and scalable

This approach mirrors how compilers, planners, and production NL2SQL systems actually work.

