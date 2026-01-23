from accelera.src.code_parallelizer.utils.code_utils import extract_loops
from accelera.src.code_parallelizer.utils.code_utils import write_loops_to_json

cpp_file = "examples/test_loops.cpp"
output_json = "loops_output.json"

print(f"Extracting loops from: {cpp_file}")
print(f"Output will be written to: {output_json}")
print("-" * 60)

loops = extract_loops(cpp_file)

print(f"\n✓ Found {len(loops)} loops:\n")

# Display each loop
for i, loop in enumerate(loops, 1):
    print(f"Loop {i}:")
    print(f"  Type: {loop.type}")
    print(f"  Lines: {loop.start_line}-{loop.end_line}")
    print("  Code preview:")

    # Show first few lines of the loop code
    code_lines = loop.code.split("\n")
    preview_lines = min(5, len(code_lines))
    for line in code_lines[:preview_lines]:
        print(f"    {line}")

    if len(code_lines) > preview_lines:
        print(f"    ... ({len(code_lines) - preview_lines} more lines)")
    print()

# Write to JSON file
if write_loops_to_json(loops, output_json):
    print(f"✓ Successfully written loops to: {output_json}")
    print("\nYou can view the JSON file with:")
    print(f"  cat {output_json}")
    print("  or")
    print(f"  python -m json.tool {output_json}")
else:
    print(f"✗ Failed to write JSON output to: {output_json}")
