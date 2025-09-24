"""
MCP Server for File Operations (Read/Write)
Protocol:
- To read a file:
  {"method": "read", "path": "file_path.txt"}
- To write a file:
  {"method": "write", "path": "file_path.txt", "content": "text to write"}
Response:
- Success: {"status": "success", "content": "file content"} (for read)
- Error: {"status": "error", "message": "error details"}
"""


import json
import sys

def read_file(path):
    try:
        with open(path, 'r') as f:
            return {"status": "success", "content": f.read()}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def write_file(path, content):
    try:
        with open(path, 'w') as f:
            f.write(content)
            return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# MCP 协议交互逻辑
if __name__ == "__main__":
    for line in sys.stdin:
        try:
            request = json.loads(line.strip())
            if request["method"] == "read":
                response = read_file(request["path"])
            elif request["method"] == "write":
                response = write_file(request["path"], request["content"])
            else:
                response = {"status": "error", "message": "Unknown method"}
            print(json.dumps(response), flush=True)
        except Exception as e:
            print(json.dumps({"status": "error", "message": str(e)}), flush=True)