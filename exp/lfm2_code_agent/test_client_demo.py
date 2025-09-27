"""Demo script showing LFM client working with tools."""

from client import LFMClient


def main():
    # Initialize client
    client = LFMClient()

    print("\n" + "="*60)
    print("LFM Client Demo - Tool Calling")
    print("="*60)

    # Test 1: File creation and reading
    print("\n🧪 Test 1: File operations with tools")
    response = client.chat_completion([
        {"role": "user", "content": "Use write_file to create a file called demo.txt with content 'This is a demo file created by LFM!' then use read_file to read it back"}
    ], max_turns=5, verbose=True)

    print(f"\nFinal response: {response['choices'][0]['message']['content']}")
    print(f"Tool calls made: {len(response['tool_calls'])}")
    print(f"Turns taken: {response['turns']}")

    # Test 2: Reading existing file
    print("\n" + "="*60)
    print("\n🧪 Test 2: Reading existing code file")
    response = client.chat_completion([
        {"role": "user", "content": "Read the code_tools.py file and summarize what functions it contains"}
    ], max_turns=3, verbose=True)

    print(f"\nFinal response: {response['choices'][0]['message']['content']}")

    # Test 3: Simple chat without tools
    print("\n" + "="*60)
    print("\n🧪 Test 3: Simple chat (no tools needed)")
    response = client.chat_completion([
        {"role": "user", "content": "Explain what an AI assistant is in one sentence"}
    ], max_turns=1, verbose=True)

    print(f"\nFinal response: {response['choices'][0]['message']['content']}")

    print("\n✅ Demo completed successfully!")


if __name__ == "__main__":
    main()