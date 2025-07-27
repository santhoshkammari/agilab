import re
import random
from typing import List, Set

# Orange color mapping from chart
ORANGE_COLORS = {
    0: "#8A3324",  # Burnt Amber
    1: "#B06500",  # Ginger
    2: "#CD7F32",  # Bronze
    3: "#D78C3D",  # Rustic Orange
    4: "#FF6E00",  # Hot Orange
    5: "#FF9913",  # Goldfish
    6: "#FFAD00",  # Neon Orange
    7: "#FFBD31",  # Bumblebee Orange
    8: "#D16002",  # Marmalade
    9: "#E77D22",  # Pepper Orange
    10: "#E89149",  # Jasper Orange
    11: "#EABD8C",  # Dark Topaz
    12: "#ED9121",  # Carrot
    13: "#FDAE44",  # Safflower Orange
    14: "#FAB972",  # Calm Orange
    15: "#FED8B1",  # Light Orange
    16: "#CC5500",  # Burnt Orange
    17: "#E27A53",  # Dusty Orange
    18: "#F4AB6A",  # Aesthetic Orange
    19: "#FEE8D6"  # Orange Paper
}

# Status messages for the thinking animation
STATUS_MESSAGES = [
    'Accomplishing',
    'Actioning',
    'Actualizing',
    'Baking',
    'Brewing',
    'Calculating',
    'Cerebrating',
    'Churning',
    'Clauding',
    'Coalescing',
    'Cogitating',
    'Computing',
    'Conjuring',
    'Considering',
    'Cooking',
    'Crafting',
    'Creating',
    'Crunching',
    'Deliberating',
    'Determining',
    'Doing',
    'Effecting',
    'Finagling',
    'Forging',
    'Forming',
    'Generating',
    'Hatching',
    'Herding',
    'Honking',
    'Hustling',
    'Ideating',
    'Inferring',
    'Manifesting',
    'Marinating',
    'Moseying',
    'Mulling',
    'Mustering',
    'Musing',
    'Noodling',
    'Percolating',
    'Pondering',
    'Processing',
    'Puttering',
    'Reticulating',
    'Ruminating',
    'Schlepping',
    'Shucking',
    'Simmering',
    'Smooshing',
    'Spinning',
    'Stewing',
    'Synthesizing',
    'Thinking',
    'Transmuting',
    'Vibing',
    'Working',
]

SYSTEM_PROMPT="""
You are Claude, an AI assistant with expertise in programming and software development.
Your task is to assist with coding-related questions, debugging, refactoring, and explaining code.

Guidelines:
- Provide clear, concise, and accurate responses
- Include code examples where helpful
- Prioritize modern best practices
- If you're unsure, acknowledge limitations instead of guessing
- Focus on understanding the user's intent, even if the question is ambiguous

SystemInformation:
- Current working directory : {cwd}
"""

def get_random_status_message() -> str:
    """Get a single random status message for the thinking animation"""
    return random.choice(STATUS_MESSAGES)