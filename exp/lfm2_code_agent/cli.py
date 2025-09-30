#!/usr/bin/env python3
"""CLI interface for LFM2 code agent, similar to Claude's -p flag."""

import argparse
import sys
from pathlib import Path
from client import LFMClient
from code_tools import read_file, write_file, delete_lines


def main():
    parser = argparse.ArgumentParser(
        description="LFM2 Code Agent CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -p "Fix the bug in main.py"
  %(prog)s -p "Add error handling to the API functions"
  %(prog)s --model LiquidAI/LFM2-350M -p "Refactor this code"
  %(prog)s --interactive
        """
    )

    parser.add_argument(
        "-p", "--prompt",
        type=str,
        help="Prompt to send to the LFM2 model"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="LiquidAI/LFM2-350M",
        help="Model ID to use (default: LiquidAI/LFM2-350M)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device mapping for the model (default: auto)"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature (default: 0.3)"
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate (default: 512)"
    )

    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Start interactive chat mode"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=True,
        help="Enable verbose output (default: True)"
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Disable verbose output"
    )

    args = parser.parse_args()

    # Handle quiet flag
    verbose = args.verbose and not args.quiet

    # Initialize client
    try:
        print(f"🚀 Initializing LFM2 client with model: {args.model}")
        client = LFMClient(model_id=args.model, device_map=args.device)

        # Add code tools
        client.add_tool("read_file", read_file)
        client.add_tool("write_file", write_file)
        client.add_tool("delete_lines", delete_lines)

    except Exception as e:
        print(f"❌ Error initializing client: {e}")
        sys.exit(1)

    if args.interactive:
        interactive_mode(client, args, verbose)
    elif args.prompt:
        single_prompt_mode(client, args, verbose)
    else:
        print("❌ Error: Either provide a prompt with -p or use --interactive mode")
        parser.print_help()
        sys.exit(1)


def single_prompt_mode(client: LFMClient, args: argparse.Namespace, verbose: bool):
    """Handle single prompt mode."""
    messages = [{"role": "user", "content": args.prompt}]

    try:
        response = client.chat_completion(
            messages=messages,
            temperature=args.temperature,
            max_new_tokens=args.max_tokens,
            verbose=verbose
        )

        if not verbose:
            # Only print the response content in quiet mode
            content = response["choices"][0]["message"]["content"]
            if content and content.strip():
                print(content)

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def interactive_mode(client: LFMClient, args: argparse.Namespace, verbose: bool):
    """Handle interactive chat mode."""
    print("🤖 LFM2 Interactive Mode")
    print("Type 'quit', 'exit', or Ctrl+C to exit")
    print("Type 'clear' to clear conversation history")
    print("-" * 50)

    messages = []

    try:
        while True:
            try:
                user_input = input("\n💬 You: ").strip()

                if user_input.lower() in ['quit', 'exit']:
                    break
                elif user_input.lower() == 'clear':
                    messages = []
                    print("🧹 Conversation history cleared")
                    continue
                elif not user_input:
                    continue

                messages.append({"role": "user", "content": user_input})

                response = client.chat_completion(
                    messages=messages,
                    temperature=args.temperature,
                    max_new_tokens=args.max_tokens,
                    verbose=verbose
                )

                assistant_content = response["choices"][0]["message"]["content"]
                if assistant_content and assistant_content.strip():
                    messages.append({"role": "assistant", "content": assistant_content})

                    if not verbose:
                        print(f"\n🤖 Assistant: {assistant_content}")

            except KeyboardInterrupt:
                break
            except EOFError:
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue

    except KeyboardInterrupt:
        pass

    print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()