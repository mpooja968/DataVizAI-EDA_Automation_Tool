import ollama

response = ollama.chat(model="llama3:8b", messages=[{"role": "user", "content": "Hello, how are you?"}])
print(response["message"]["content"])
