import groq

API_KEY = "give_your_api_key"

client = groq.Client(api_key=API_KEY)

def chat_with_groq(prompt):
    response = client.chat.completions.create(
        model="llama3-8b-8192",  # Correct model name
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

user_input = input("Enter your prompt: ")
response = chat_with_groq(user_input)
print("\nResponse:", response)

