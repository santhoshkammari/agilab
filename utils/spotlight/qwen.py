import subprocess
import json
from typing import List, Iterator, Dict, Any

# Default directories to include for system access
INCLUDE_DIRECTORIES = ['/home', '/tmp', '/etc', '/var', '/opt']

def qwen(args: List[str]) -> Iterator[Dict[str, Any]]:
    dirs = ",".join(INCLUDE_DIRECTORIES)
    cmd = ['qwen', '-y', '--include-directories',dirs,'--output-format', 'stream-json']
    cmd.extend(args)

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        bufsize=1
    )

    for stdout_line in iter(process.stdout.readline, ''):
        line = stdout_line.strip()
        if line:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Skip invalid JSON lines
                continue

    process.stdout.close()
    return_code = process.wait()

    if return_code != 0:
        stderr_output = process.stderr.read() if process.stderr else ""
        raise subprocess.CalledProcessError(return_code, cmd, stderr=stderr_output)


if __name__ == "__main__":
    # Example with system directory access (using default INCLUDE_DIRECTORIES)
    for chunk in qwen(['-p', 'check if /home/ntlpt24/agilab directory exists and list its contents using bash commands']):
        print(f"----Chunkjsonreceived-----\n{chunk}")
