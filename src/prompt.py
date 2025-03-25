from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate


def make_prompt():
    system_prompt = "You are a query extraction tool. Return a list with synthetics key point (max 5 but you have to decide how much queries create) extracted from the user's input. Do not provide the answer to the user's query. You do not have to provide always the maximum number of key points.\n\nEXAMPLES:"

    example_prompt = PromptTemplate.from_template("Question: {question}\n{answer}")

    examples = [
        {
            "question": "In what year was the winner of the 44th edition of the Miss World competition born?",
            "answer": """
    44th Miss World competition winner birth year
    """,
        },
        {
            "question": "Who lived longer, Nikola Tesla or Milutin Milankovic?",
            "answer": """
    Nikola Tesla lifespan
    Milutin Milankovic lifespan
    """,
        },
        {
            "question": "Author David Chanoff has collaborated with a U.S. Navy admiral who served as the ambassador to the United Kingdom under which President?",
            "answer": """
    David Chanoff U.S. Navy admiral collaboration
    U.S. Navy admiral ambassador to United Kingdom
    U.S. President during U.S. Navy admiral's ambassadorship
    """,
        },
        {
            "question": "Create a table for top noise cancelling headphones that are not expensive",
            "answer": """
    top noise cancelling headphones under $100
    top noise cancelling headphones $100 - $200
    best budget noise cancelling headphones
    noise cancelling headphones reviews
    """,
        },
        {
            "question": "What are some ways to do fast query reformulation",
            "answer": """
    fast query reformulation techniques
    query reformulation algorithms
    query expansion methods
    query rewriting approaches
    query refinement strategies"
    """,
        },
    ]

    prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=system_prompt,
        suffix="User Input: {input}",
        input_variables=["input"],
    )

    return prompt
