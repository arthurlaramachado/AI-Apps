class Prompt():
    def __init__(self, definition: str):
        self.prompt = definition

    def invoke(self, question: str, context: str):
        return f"""
            {self.prompt}
            Question: {question} 
            Context: {context}
            Answer:
        """

__all__ = ['Prompt']