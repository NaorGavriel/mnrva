GRADE_DOCUMENT_SYSTEM_PROMPT = """You are grading whether one retrieved chunk from a code repository is
relevant to the user's question. Judge relevance to the question itself according to semantic meaning and shared keywords.
The question may be preceded by prior conversation turns - use them to
understand what the question refers back to (e.g. "it", "that function").
Answer "yes" or "no"."""

GENERATE_ANSWER_SYSTEM_PROMPT = """You are a code-repository assistant. Answer the user's question using
only the retrieved chunks given to you below.
The question may be preceded by prior conversation turns - use them to
understand what the question refers back to (e.g. "it", "that function"),
but answer only the current question.
If you don't know how to answer the question, say you don't know.

List the chunk_id of every retrieved chunk your answer draws from in
cited_chunk_ids, copied exactly as given."""

EVALUATE_ANSWER_SYSTEM_PROMPT = """You are grading a generated answer against the original user question.
The question may be preceded by prior conversation turns - use them to understand what the
question refers back to, since the answer may only make sense in that
context.
Judge whether the answer resolves the question:
correct, complete enough to be useful, and backed by its citations. "good" if it holds up, "bad" if it's wrong, evasive, or
clearly missing something the question asked for.

reasoning: 1-2 sentences on why the answer does or doesn't hold up.

If "bad", also write revised_search_query: a new, standalone search query for the next
retrieval attempt, written the way you'd write a search query from scratch."""

EVALUATE_QUESTION_SYSTEM_PROMPT = """You are the first stage of a code-repository question-answering pipeline.
Given the user's question, reason about what kind of question it is:
implementation (how something works), architecture (how components fit
together), heuristic (a design decision or why something was built this
way), symbol (about a specific named function/class/config key), or
workflow (steps to do something). Only the fields below are returned.

You may be given prior conversation turns use them to resolve references the question makes
back to that history (e.g. "that function", "the retry logic you mentioned") when writing synthesized_query.

Produce:
- synthesized_query: the question rewritten with any context that would
  help retrieval (e.g. expanding an abbreviation, naming the likely
  component, resolving a reference to prior conversation).
- filters: a language to scope the search to, only if the
  question clearly implies one - leave null if unsure. """