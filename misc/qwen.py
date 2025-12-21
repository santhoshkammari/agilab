import subprocess
import json
from typing import List, Iterator, Dict, Any


def qwen(args: List[str]) -> Iterator[Dict[str, Any]]:
    cmd = ['qwen','-y', '--output-format', 'stream-json'] + args

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        bufsize=1
    )

    for stdout_line in process.stdout:
        line = stdout_line.strip()
        if line:
            yield json.loads(line)

    return_code = process.wait()
    stderr_output = process.stderr.read() if process.stderr else ""

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd, stderr=stderr_output)


if __name__ == "__main__":
    for chunk in qwen(['-p', 'run bash sleep for 5 seconds, again run bash for 2 secodn adn then say what is 2+3?']):
        print(f"Received: {chunk}")
