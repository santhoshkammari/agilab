#!/usr/bin/env python3
"""
Simple print function example
"""

def simple_print(message):
    """Print a message to console"""
    print(f"Message: {message}")

def greet(name):
    """Greet someone by name"""
    print(f"Hello, {name}!")

def main():
    """Main function to demonstrate the print functions"""
    simple_print("This is a test message")
    greet("World")

if __name__ == "__main__":
    main()