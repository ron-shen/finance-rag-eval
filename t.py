from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextRecall

# Setup LLM
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

# Create metric
scorer = ContextRecall(llm=llm)

# Evaluate
result = scorer.score(
    user_input="Where is the Eiffel Tower located?",
    retrieved_contexts=["Paris", "I go to school by bus"],
    reference="The Eiffel Tower is located in Paris."
)
print(f"Context Recall Score: {result.value}")