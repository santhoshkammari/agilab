#!/usr/bin/env python3
import subprocess
import json
import re
from datetime import datetime

def run_claude_usage():
    try:
        result = subprocess.run(
            ['claude', '-p', '/usage'],
            capture_output=True,
            text=True,
            timeout=30,
            input='/usage\n'
        )

        output = result.stdout
        error = result.stderr

        if result.returncode != 0 and not output:
            return {
                "status": "error",
                "returncode": result.returncode,
                "error": error,
                "output": output,
                "note": "/usage is an in-session command, not a CLI command"
            }

        usage_data = parse_usage_output(output)
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "data": usage_data,
            "raw_output": output
        }

    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "Command timeout"}
    except FileNotFoundError:
        return {"status": "error", "message": "claude command not found"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def parse_usage_output(output):
    parsed = {}
    lines = output.strip().split('\n')

    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().lower().replace(' ', '_')
            value = value.strip()

            if value.replace(',', '').replace('.', '').isdigit():
                value = float(value.replace(',', ''))

            parsed[key] = value

    return parsed if parsed else {"raw_lines": lines}

if __name__ == "__main__":
    result = run_claude_usage()
    print(json.dumps(result, indent=2))
