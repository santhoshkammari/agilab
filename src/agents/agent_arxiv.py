from pydantic import BaseModel
from typing import List
import asyncio

class Point(BaseModel):
    text: str
    level: str

class Para(BaseModel):
    id: int
    points: List[Point]

class Section(BaseModel):
    name: str
    summary: str
    findings: List[str]

class Paper(BaseModel):
    sections: List[Section]
    tldr: str

async def paras(lm, text):
    msgs = [[{"role": "user", "content": f"Extract: {p}"}] for p in text]
    results = await lm.batch(msgs, response_format={
        "type": "json_schema",
        "json_schema": {"name": "para", "schema": Para.model_json_schema()}
    })
    return [Para.model_validate_json(r['choices'][0]['message']['content']) for r in results]

async def section(lm, name, data):
    msg = [{"role": "user", "content": f"Summarize {name}: {data}"}]
    result = await lm.batch([msg], response_format={
        "type": "json_schema",
        "json_schema": {"name": "section", "schema": Section.model_json_schema()}
    })
    return Section.model_validate_json(result[0]['choices'][0]['message']['content'])

async def paper(lm, sections):
    await lm.start()
    results = []
    for name, text in sections.items():
        p = await paras(lm, text)
        s = await section(lm, name, p)
        results.append(s)
    await lm.close()
    return results

# usage
async def main():
    lm = LM(model="gpt-4", api_base="https://api.openai.com")
    data = {
        "Abstract": ["para1", "para2"],
        "Method": ["para3", "para4", "para5"],
    }
    output = await paper(lm, data)
    print(output)
