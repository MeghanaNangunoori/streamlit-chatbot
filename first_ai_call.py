from groq import Groq

client = Groq(api_key="YOUR_GROQ_API_KEY")
history=[]
while True:

    user_qn= input("Ask me anything: ")

    history.append({"role":"user","content":user_qn})

    response=client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=history
    )
    ai_answer=response.choices[0].message.content

    history.append({"role":"assistant","content":ai_answer})

    print(ai_answer)
    