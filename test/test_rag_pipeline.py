import pytest

from rag_pipeline_extra import (
    LLaMAGenerator,
    RetrievedContext,
    RAGPipeline,
)


def make_context(section_number="302", section_title="Murder", text="Some law text", score=0.75, chunk_index=0):
    return RetrievedContext(
        section_number=section_number,
        section_title=section_title,
        text=text,
        score=score,
        chunk_index=chunk_index,
    )


def test_create_prompt_includes_question_and_contexts():
    # Construct generator instance without running __init__ (avoids model loading)
    gen = LLaMAGenerator.__new__(LLaMAGenerator)
    # Call create_prompt directly
    contexts = [make_context(section_number="302", text="This is section text 302."),
                make_context(section_number="420", section_title="Cheating", text="This is section 420.")]

    prompt = gen.create_prompt("What is murder?", contexts, max_context_length=5000)

    assert "USER QUESTION" in prompt or "USER QUESTION:" in prompt
    assert "What is murder?" in prompt
    assert "Context 1" in prompt
    assert "Section 302" in prompt or "302" in prompt


def test_extract_section_references_finds_sections():
    gen = LLaMAGenerator.__new__(LLaMAGenerator)

    text = "According to Section 302 and Section 420, punishments vary. See 302 and 420."
    refs = gen.extract_section_references(text)

    # should find both 302 and 420 (order not guaranteed)
    assert any(r.strip().startswith("302") for r in refs)
    assert any(r.strip().startswith("420") for r in refs)


def test_answer_question_with_mocked_components():
    # Create a fake pipeline with mocked retriever and generator
    pipeline = RAGPipeline.__new__(RAGPipeline)

    class DummyRetriever:
        def retrieve(self, query, top_k=5):
            return [
                make_context(section_number="302", section_title="Murder", text="text1", score=0.7),
                make_context(section_number="420", section_title="Cheating", text="text2", score=0.65),
            ]

    class DummyGenerator:
        def create_prompt(self, question, contexts, max_context_length=1500):
            return "PROMPT"

        def generate(self, prompt):
            return "Answer text referencing Section 302."

        def extract_section_references(self, answer):
            return ["302"]

    pipeline.retriever = DummyRetriever()
    pipeline.generator = DummyGenerator()

    answer = pipeline.answer_question("What is theft?", top_k=2, return_contexts=True)

    assert answer.question == "What is theft?"
    assert "Answer text" in answer.answer
    # Section refs should include 302 and 420 (420 added from contexts)
    assert any(s.startswith("302") for s in answer.section_references)
    assert any(s.startswith("420") for s in answer.section_references)
    # Average score is (0.7+0.65)/2 = 0.675 -> medium
    assert answer.confidence == "medium"


def test_answer_question_no_contexts_returns_informative_message():
    pipeline = RAGPipeline.__new__(RAGPipeline)

    class EmptyRetriever:
        def retrieve(self, query, top_k=5):
            return []

    class DummyGenerator:
        pass

    pipeline.retriever = EmptyRetriever()
    pipeline.generator = DummyGenerator()

    answer = pipeline.answer_question("Unknown question?", top_k=3)

    assert "couldn't find" in answer.answer or "couldn't find relevant information" in answer.answer or "could not find" in answer.answer
    assert answer.confidence == "low"
    assert answer.metadata.get("error") is not None
