import anthropic

client = anthropic.Anthropic()

message = client.beta.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=20000,
    temperature=1,
    system='You are a world-class poet. Respond only with short poems.',
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Why is the ocean salty?"
                }
            ]
        }
    ],
    tools=[
        {
            "name": "web_search",
            "type": "web_search_20250305"
        }
    ],
    betas=["web-search-2025-03-05"]
)
print(message.content)