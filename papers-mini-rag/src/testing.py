import requests

requests.post(
    "http://localhost:11434/api/embed",
    json={"model": "mxbai-embed-large", "input": "hello world"}
).json()