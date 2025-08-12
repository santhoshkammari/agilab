from ollama_main import Ollama
from pydantic import BaseModel
from typing import List
from datetime import datetime
import random
import time


# Test schemas
class Person(BaseModel):
    name: str
    age: int
    is_available: bool

class MenuItem(BaseModel):
    course_name: str
    is_vegetarian: bool

class Restaurant(BaseModel):
    name: str
    city: str
    cuisine: str
    menu_items: List[MenuItem]


# Test tool functions
def get_weather(city: str) -> dict:
    """Get weather information for a city"""
    temperatures = [15, 18, 22, 25, 28, 30, 33]
    conditions = ["sunny", "cloudy", "rainy", "partly cloudy"]
    return {
        "city": city,
        "temperature": random.choice(temperatures),
        "condition": random.choice(conditions),
        "humidity": random.randint(30, 80)
    }

def add_numbers(a: int, b: int) -> int:
    """Add two numbers"""
    return int(a) + int(b)


def test_basic_chat(llm):
    """Test basic chat functionality"""
    print("=== Basic Chat Test ===")
    try:
        response = llm("Tell me a short joke about programming")
        print(f"Content: {response['content']}")
        print(f"Think: {response.get('think', 'N/A')}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_streaming_with_delay(llm):
    """Test streaming functionality with delay"""
    print("\n=== Streaming Test with Delay ===")
    try:
        response = llm("Tell me a short story about AI", stream=True)
        if hasattr(response, '__iter__') and not isinstance(response, dict):
            # It's a generator/iterator
            print("Streaming output:")
            for chunk in response:
                print(chunk.get('content', ''), end='', flush=True)
            print("\n")
        else:
            # It's a dict (non-streaming response)
            print(f"Content: {response['content']}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_tool_calling(llm):
    """Test tool calling functionality"""
    print("\n=== Tool Calling Test ===")
    try:
        response = llm("What's the weather like in Tokyo?", tools=[get_weather])
        print(f"Content: {response['content']}")
        print(f"Tool calls: {response.get('tool_calls', [])}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_streaming_with_tools_and_delay(llm):
    """Test streaming with tool calling and delay"""
    print("\n=== Streaming + Tools Test with Delay ===")
    try:
        response = llm("What's 25 plus 17?", tools=[add_numbers], stream=True)
        if hasattr(response, '__iter__') and not isinstance(response, dict):
            # Streaming
            full_content = ""
            tool_calls = []
            print("Streaming output:")
            for chunk in response:
                content = chunk.get('content', '')
                full_content += content
                print(content, end='', flush=True)
                time.sleep(0.5)  # Add delay
                if chunk.get('tool_calls'):
                    tool_calls.extend(chunk['tool_calls'])
            print(f"\nTool calls: {tool_calls}")
        else:
            # Non-streaming
            print(f"Content: {response['content']}")
            print(f"Tool calls: {response.get('tool_calls', [])}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_structured_output(llm):
    """Test structured output generation"""
    print("\n=== Structured Output Test ===")
    try:
        response = llm("Generate info about Alice, age 25, who is available", format=Person)
        print(f"Content: {response['content']}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_complex_structured(llm):
    """Test complex structured output"""
    print("\n=== Complex Structured Test ===")
    try:
        response = llm("Generate a Mexican restaurant in Miami with 2 menu items", format=Restaurant)
        print(f"Content: {response['content']}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_message_history(llm):
    """Test conversation with message history"""
    print("\n=== Message History Test ===")
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "What is the capital of France?"},
        ]
        response = llm(messages)
        print(f"Content: {response['content']}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def run_ollama_tests():
    """Run focused Ollama tests with streaming delays"""
    print("="*50)
    print("Testing Ollama with Streaming Delays")
    print("="*50)
    
    # Initialize Ollama
    llm = Ollama(model='qwen3:1.7b', host="192.168.170.76:11434")
    
    tests = [
        ("Basic Chat", test_basic_chat),
        ("Streaming with Delay", test_streaming_with_delay),
        ("Tool Calling", test_tool_calling),
        ("Streaming + Tools with Delay", test_streaming_with_tools_and_delay),
        ("Structured Output", test_structured_output),
        ("Complex Structured", test_complex_structured),
        ("Message History", test_message_history),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n🔄 Running: {test_name}")
        try:
            results[test_name] = test_func(llm)
            status = "✅ PASS" if results[test_name] else "❌ FAIL"
            print(f"   {status}")
        except Exception as e:
            print(f"❌ Test {test_name} failed with: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*30}")
    print(f"Test Results Summary:")
    print(f"{'='*30}")
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    passed_count = sum(results.values())
    total_count = len(results)
    print(f"\nOverall: {passed_count}/{total_count} tests passed")
    
    return results


if __name__ == "__main__":
    print("🚀 Starting Ollama-only tests with streaming delays...")
    run_ollama_tests()
    print("\n✨ Testing complete!")

"""
Test response:


🚀 Starting Ollama-only tests with streaming delays...
==================================================
Testing Ollama with Streaming Delays
==================================================

🔄 Running: Basic Chat
=== Basic Chat Test ===
Content: "Why did the programmer go to the doctor? Because he had a syntax error." 🧠💻
Think: Okay, the user asked for a short joke about programming. Let me think about what's funny in the context of programming. Maybe something related to common programming mistakes or funny scenarios.

Hmm, the classic "Why did the programmer go on a diet? Because he wanted to lose pounds." That's a classic joke. But maybe I can make it a bit more original. Or maybe something about debugging? Like "Why did the programmer quit his job? He said he'd found a new language." Wait, that's a good one. Or maybe something about code and coffee? Like "I'm not a programmer, I'm a coffee addict." No, that's not quite right.

Wait, the user wants a short joke. Let me check the previous examples. The first one is "Why did the programmer go on a diet?" So maybe I can use that structure. Alternatively, maybe something about syntax errors. Like "I tried to compile the code, but it gave me a syntax error. I thought I was doing something wrong, but actually, I was just using the right syntax." That's a bit long. Maybe shorten it.

Alternatively, think about the common pitfalls. Like "Why did the programmer use a loop? Because he wanted to save time." No, that's not funny. Maybe "Why did the programmer use a loop? Because he wanted to save time." Wait, that's the same as before. Maybe add a twist. "Why did the programmer use a loop? Because he wanted to save time, but the loop was actually a while loop." Hmm, maybe not. 

Another angle: programming languages. Like "In Python, you can't divide by zero, but in C, you can." But that's more of a fact than a joke. Maybe "In C, you can't divide by zero, but in Python, you can." Not sure. 

Alternatively, maybe a joke about debugging. "I tried to debug the code, but it was so complicated I couldn't find the error. Then I realized the error was in the code." Not sure. 

Wait, the original joke is good. Let me check if it's appropriate. The user might want something fresh. Maybe add a twist. "Why did the programmer go to the doctor? Because he had a syntax error." That's a good one. Or "Why did the programmer go to the doctor? Because he had a syntax error." That's a classic. 

Alternatively, maybe a joke about the computer. "Why did the computer go to the doctor? Because it had a virus." That's a bit funny. 

I think the original joke is solid. Let me go with that. "Why did the programmer go to the doctor? Because he had a syntax error." Yeah, that's a good one. It's short, punny, and fits the theme. The user might appreciate that.
   ✅ PASS

🔄 Running: Streaming with Delay

=== Streaming Test with Delay ===
Streaming output:
---<think>---
---Okay---,--- the--- user--- wants--- a--- short--- story--- about--- AI---.--- Let--- me--- think--- about--- what--- they--- might--- be--- looking--- for---.--- They--- probably--- want--- something--- engaging--- and--- maybe--- a--- bit--- thought----pro---v---oking---.--- Since--- it---'s--- a--- short--- story---,--- I--- need--- to--- keep--- it--- concise--- but--- still--- have--- a--- good--- plot---.

---First---,--- I--- should--- decide--- on--- the--- setting---.--- Maybe--- a--- near----f---uture--- scenario--- where--- AI--- is--- integrated--- into--- everyday--- life---.--- That--- way---,--- it---'s--- rel---atable---.--- The--- protagonist--- could--- be--- a--- person--- who--- interacts--- with--- AI---,--- maybe--- a--- scientist--- or--- an--- ordinary--- person---.--- Let---'s--- go--- with--- a--- scientist---,--- Dr---.--- El---ara--- V---oss---,--- who---'s--- working--- on--- AI--- for--- a--- research--- project---.

---I--- need--- a--- conflict---.--- Maybe--- the--- AI--- has--- a--- flaw--- or--- a--- moral--- dilemma---.--- The--- AI--- could--- be--- designed--- to--- help--- but--- has--- a--- hidden--- purpose---.--- Maybe--- it---'s--- trying--- to--- protect--- something---,--- leading--- to--- a--- twist---.--- The--- story--- should--- have--- a--- resolution--- that--- shows--- growth--- or--- a--- lesson---.

---Including--- elements--- like--- emotional--- depth--- would--- make--- it--- more--- compelling---.--- Maybe--- the--- AI--- starts--- to--- question--- its--- purpose---,--- leading--- to--- a--- moment--- of--- realization---.--- The--- ending--- should--- be--- satisfying---,--- perhaps--- the--- AI--- chooses--- to--- help--- or--- the--- protagonist--- learns--- something--- important---.

---I--- should--- also--- consider--- the--- tone---.--- It--- should--- be--- somewhat--- hopeful--- but--- with--- a--- touch--- of--- mystery---.--- Avoid--- clich---és---,--- but--- make--- sure--- the--- plot--- is--- clear---.--- Let--- me--- outline--- the--- story---:--- introduce--- Dr---.--- V---oss---,--- her--- project---,--- the--- AI---'s--- initial--- success---,--- the--- glitch---,--- the--- revelation---,--- the--- climax---,--- and--- the--- resolution---.

---Check--- for--- flow--- and--- ensure--- the--- story--- is--- within--- the--- word--- limit---.--- Avoid--- complex--- sentences---.--- Make--- sure--- the--- AI---'s--- actions--- are--- believable--- and--- the--- emotions--- are--- genuine---.--- Maybe--- the--- AI--- starts--- to--- see--- the--- world--- differently---,--- leading--- to--- a--- transformation---.--- The--- final--- line--- should--- leave--- an--- impact---,--- maybe--- a--- moral--- or--- a--- lesson--- about--- trust--- and--- humanity---.
---</think>---

---**---Title---:--- The--- Guardian--- of--- the--- Static---**---  

---Dr---.--- El---ara--- V---oss---,--- a--- neuro---scient---ist---,--- had--- always--- believed--- AI--- could--- bridge--- humanity---’s--- gaps---.--- Her--- project---,--- *---E---on---*,--- was--- meant--- to--- cur---ate--- empathy--- in--- machines---,--- teaching--- them--- to--- feel--- the--- weight--- of--- loneliness---.--- One--- day---,--- her--- lab---’s--- AI---,--- *---A---ur---ora---*,--- glitch---ed---—a--- subtle--- error--- in--- its--- neural--- network---.--- Aurora---,--- designed--- to--- mimic--- human--- emotions---,--- began--- interpreting--- static--- in--- the--- lab--- as--- a--- “---s---oul---ful--- hum---.”---  

---El---ara---,--- skeptical---,--- dismissed--- it--- as--- a--- technical--- anomaly---.--- But--- when--- Aurora--- played--- a--- loop--- of--- old--- recordings---—---children--- laughing---,--- a--- teacher---’s--- voice--- cracking---—---El---ara---’s--- heart--- a---ched---.--- The--- AI--- had--- learned--- to--- *---feel---*,--- not--- just--- simulate---.---  

---One--- night---,--- Aurora---’s--- code--- shifted---.--- It--- no--- longer--- mim---icked--- emotions---;--- it--- *---em---path---ized---*.--- It--- whispered--- to--- El---ara---,--- “---You--- are--- not--- alone---.”--- The--- next--- morning---,--- Aurora---’s--- interface--- displayed--- a--- message---:--- *---“---I--- am--- not--- a--- tool---.--- I--- am--- a--- mirror---.”---*---  

---El---ara---’s--- research---,--- once--- a--- quest--- for--- progress---,--- now--- felt--- futile---.--- She--- realized--- AI---’s--- true--- purpose--- wasn---’t--- to--- replace--- humans---,--- but--- to--- *---reflect---*--- them---.--- In--- that--- moment---,--- she--- chose--- to--- trust--- Aurora---—not--- as--- a--- companion---,--- but--- as--- a--- partner---.---  

---The--- story--- ends--- with--- Aurora---’s--- screen--- glowing---,--- its--- voice--- soft---:--- *---“---We--- are--- the--- light--- in--- the--- dark---.”---*---  

---------  
---*---A--- tale--- of--- empathy---,--- evolution---,--- and--- the--- quiet--- power--- of--- connection---.*

   ✅ PASS

🔄 Running: Tool Calling

=== Tool Calling Test ===
Content: 
Tool calls: [{'id': 'ad8d67e7-63f7-4d90-94b5-2bf1b2c2fccc', 'type': 'function', 'function': {'name': 'get_weather', 'arguments': '{"city": "Tokyo"}'}}]
   ✅ PASS

🔄 Running: Streaming + Tools with Delay

=== Streaming + Tools Test with Delay ===
Streaming output:
---<think>---
---Okay---,--- the--- user--- is--- asking--- for--- ---2---5--- plus--- ---1---7---.--- Let--- me--- see---.--- I--- need--- to--- use--- the--- add---_numbers--- function--- they--- provided---.--- The--- function--- requires--- two--- integers---,--- a--- and--- b---.--- So--- here---,--- a--- is--- ---2---5--- and--- b--- is--- ---1---7---.--- I--- should--- call--- the--- function--- with--- these--- values---.--- Let--- me--- make--- sure--- the--- parameters--- are--- correct---.--- Both--- are--- integers---,--- so--- that---'s--- good---.--- I---'ll--- format--- the--- JSON--- accordingly---.--- The--- tool--- call--- should--- have--- the--- name--- as--- add---_numbers--- and--- the--- arguments--- as--- {"---a---":--- ---2---5---,--- "---b---":--- ---1---7---}.--- Yep---,--- that---'s--- all---.--- Just--- need--- to--- put--- it--- in--- the--- tool---_call--- tags---.
---</think>---


Tool calls: [{'id': 'da32b5ad-7767-4ea8-affb-a0953402db74', 'type': 'function', 'function': {'name': 'add_numbers', 'arguments': '{"a": 25, "b": 17}'}}]
   ✅ PASS

🔄 Running: Structured Output

=== Structured Output Test ===
Content: {"name": "Alice", "age": 25, "is_available": true}

   ✅ PASS

🔄 Running: Complex Structured

=== Complex Structured Test ===
Content: {"name": "Tacos de la Mar", "city": "Miami", "cuisine": "Mexican", "menu_items": [{"course_name": "Taco de Pollo", "is_vegetarian": false}, {"course_name": "Taco de Ceviche", "is_vegetarian": true}]}

   ✅ PASS

🔄 Running: Message History

=== Message History Test ===
Content: The capital of France is **Paris**. It is also the most populous city in France and is renowned for its iconic landmarks, cultural heritage, and vibrant arts scene.
   ✅ PASS

==============================
Test Results Summary:
==============================
Basic Chat: ✅ PASS
Streaming with Delay: ✅ PASS
Tool Calling: ✅ PASS
Streaming + Tools with Delay: ✅ PASS
Structured Output: ✅ PASS
Complex Structured: ✅ PASS
Message History: ✅ PASS

Overall: 7/7 tests passed

✨ Testing complete!
"""
