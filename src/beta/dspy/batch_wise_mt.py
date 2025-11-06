import aspy as a

# Setup
lm = a.LM(api_base="http://192.168.170.76:8000")
a.configure(lm=lm)

# Simple test of the new batch system
from mt_ext import EntityExtractor

extractor = EntityExtractor()

# Test with a few examples
test_inputs = [
    {"text": "All Scheduled Commercial Banks (including Small Finance Banks and excluding RRBs)"},
    {"text": "The chairman / Managing Director / Chief Executive Officers All Scheduled Commercial Banks"},
    {"text": "Regional Rural Banks excluding cooperative banks"}
]

print("Testing new batch system...")
print("=" * 50)

# Test single execution first
print("Single execution:")
result = extractor(text=test_inputs[0]["text"])
print(f"Including: {result.including}")
print(f"Excluding: {result.excluding}")
print(f"General: {result.general}")
print()

# Test batch execution
print("Batch execution:")
results = extractor.batch(test_inputs)

for i, result in enumerate(results):
    print(f"Input {i+1}: {test_inputs[i]['text']}")
    print(f"Including: {result.including}")
    print(f"Excluding: {result.excluding}")
    print(f"General: {result.general}")
    print("---")

print("Test completed!")