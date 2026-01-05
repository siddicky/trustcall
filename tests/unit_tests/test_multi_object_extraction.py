"""Tests for multi-object extraction functionality."""

import uuid
from typing import Any, Dict, List, Optional

import pytest
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import SimpleChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel, Field

from trustcall import create_multi_object_extractor


class FakeMultiExtractionModel(SimpleChatModel):
    """Fake Chat Model wrapper for testing multi-object extraction."""

    responses: List[AIMessage] = []
    i: int = 0
    tools: list = []

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        return "fake response"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Use global index tracking across all instances
        if not hasattr(FakeMultiExtractionModel, "_global_index"):
            FakeMultiExtractionModel._global_index = 0

        idx = FakeMultiExtractionModel._global_index
        FakeMultiExtractionModel._global_index += 1
        message = self.responses[idx % len(self.responses)]
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Use global index tracking across all instances
        if not hasattr(FakeMultiExtractionModel, "_global_index"):
            FakeMultiExtractionModel._global_index = 0

        idx = FakeMultiExtractionModel._global_index
        FakeMultiExtractionModel._global_index += 1
        message = self.responses[idx % len(self.responses)]
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "fake-multi-chat-model"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"key": "fake"}

    def bind_tools(self, tools: list, **kwargs: Any) -> "FakeMultiExtractionModel":
        """Bind tools to the model."""
        from langchain_core.utils.function_calling import convert_to_openai_tool

        tools = [convert_to_openai_tool(t) for t in tools]
        # Share responses across all bound instances
        return FakeMultiExtractionModel(
            responses=self.responses,
            tools=tools,
            i=self.i,
            **kwargs,
        )


class Person(BaseModel):
    """A person schema for testing."""

    name: str = Field(description="Person's full name")
    age: int = Field(description="Person's age")
    occupation: str = Field(description="Person's occupation")


class CustomIdentifier(BaseModel):
    """Custom identification schema."""

    person_name: str = Field(description="Name of the person")
    mentioned_in: str = Field(description="Where they were mentioned")


class Address(BaseModel):
    """Address schema."""

    city: str = Field(description="City name")
    zipcode: str = Field(description="Zip code")


class UserWithAddress(BaseModel):
    """User with address schema."""

    name: str = Field(description="User's name")
    address: Address = Field(description="User's address")


@pytest.mark.asyncio
async def test_multi_object_extraction_basic():
    """Test basic multi-object extraction with two persons."""
    # Reset the global index for this test
    if hasattr(FakeMultiExtractionModel, "_global_index"):
        FakeMultiExtractionModel._global_index = 0

    # Phase 1: Identification response
    identification_response = AIMessage(
        content="Identified 2 people",
        tool_calls=[
            {
                "id": f"identify_{uuid.uuid4()}",
                "name": "MultipleObjectIdentifiers",
                "args": {
                    "objects": [
                        {
                            "name": "Alice",
                            "distinguishing_context": "30 year old engineer",
                        },
                        {
                            "name": "Bob",
                            "distinguishing_context": "25 year old designer",
                        },
                    ]
                },
            }
        ],
    )

    # Phase 2: Individual extraction responses
    alice_response = AIMessage(
        content="Extracted Alice",
        tool_calls=[
            {
                "id": f"person_{uuid.uuid4()}",
                "name": "Person",
                "args": {"name": "Alice", "age": 30, "occupation": "engineer"},
            }
        ],
    )

    bob_response = AIMessage(
        content="Extracted Bob",
        tool_calls=[
            {
                "id": f"person_{uuid.uuid4()}",
                "name": "Person",
                "args": {"name": "Bob", "age": 25, "occupation": "designer"},
            }
        ],
    )

    # Create model with responses
    model = FakeMultiExtractionModel(
        responses=[identification_response, alice_response, bob_response]
    )

    extractor = create_multi_object_extractor(
        model, target_schema=Person, max_objects=5
    )

    result = await extractor.ainvoke(
        "Alice is 30 and works as an engineer. Bob is 25 and is a designer."
    )

    # Verify results
    assert len(result["responses"]) == 2
    assert result["identification_count"] == 2

    # Check extracted persons
    alice = next((p for p in result["responses"] if p.name == "Alice"), None)
    bob = next((p for p in result["responses"] if p.name == "Bob"), None)

    assert alice is not None
    assert alice.age == 30
    assert alice.occupation == "engineer"

    assert bob is not None
    assert bob.age == 25
    assert bob.occupation == "designer"


@pytest.mark.asyncio
async def test_multi_object_extraction_with_validation():
    """Test multi-object extraction with validation (no retry in parallel for now)."""
    # Reset the global index for this test
    if hasattr(FakeMultiExtractionModel, "_global_index"):
        FakeMultiExtractionModel._global_index = 0

    # Reset the global index for this test
    if hasattr(FakeMultiExtractionModel, "_global_index"):
        FakeMultiExtractionModel._global_index = 0

    # Phase 1: Identification response
    identification_response = AIMessage(
        content="Identified 1 person",
        tool_calls=[
            {
                "id": f"identify_{uuid.uuid4()}",
                "name": "MultipleObjectIdentifiers",
                "args": {
                    "objects": [
                        {
                            "name": "Charlie",
                            "distinguishing_context": "software developer",
                        }
                    ]
                },
            }
        ],
    )

    # Phase 2: Correct extraction (no error this time)
    charlie_response = AIMessage(
        content="Extracted Charlie",
        tool_calls=[
            {
                "id": f"person_{uuid.uuid4()}",
                "name": "Person",
                "args": {
                    "name": "Charlie",
                    "age": 35,  # Correct type
                    "occupation": "software developer",
                },
            }
        ],
    )

    model = FakeMultiExtractionModel(
        responses=[identification_response, charlie_response]
    )

    extractor = create_multi_object_extractor(
        model, target_schema=Person, max_objects=5
    )

    result = await extractor.ainvoke(
        "Charlie is thirty-five and is a software developer"
    )

    # Verify results
    assert len(result["responses"]) == 1
    assert result["identification_count"] == 1

    charlie = result["responses"][0]
    assert charlie.name == "Charlie"
    assert charlie.age == 35
    assert charlie.occupation == "software developer"


@pytest.mark.asyncio
async def test_multi_object_extraction_custom_identifier():
    """Test multi-object extraction with custom identification schema."""
    # Reset the global index for this test
    if hasattr(FakeMultiExtractionModel, "_global_index"):
        FakeMultiExtractionModel._global_index = 0

    # Phase 1: Identification with custom schema
    identification_response = AIMessage(
        content="Identified people",
        tool_calls=[
            {
                "id": f"identify_{uuid.uuid4()}",
                "name": "MultipleObjectIdentifiers",
                "args": {
                    "objects": [
                        {
                            "person_name": "Diana",
                            "mentioned_in": "first paragraph",
                        }
                    ]
                },
            }
        ],
    )

    # Phase 2: Extraction response
    diana_response = AIMessage(
        content="Extracted Diana",
        tool_calls=[
            {
                "id": f"person_{uuid.uuid4()}",
                "name": "Person",
                "args": {"name": "Diana", "age": 28, "occupation": "teacher"},
            }
        ],
    )

    model = FakeMultiExtractionModel(
        responses=[identification_response, diana_response]
    )

    extractor = create_multi_object_extractor(
        model,
        target_schema=Person,
        identification_schema=CustomIdentifier,
        max_objects=5,
    )

    result = await extractor.ainvoke("Diana is a 28 year old teacher.")

    # Verify results
    assert len(result["responses"]) == 1
    assert result["identification_count"] == 1

    diana = result["responses"][0]
    assert diana.name == "Diana"
    assert diana.age == 28
    assert diana.occupation == "teacher"


@pytest.mark.asyncio
async def test_multi_object_extraction_max_objects_limit():
    """Test that max_objects limit is enforced."""
    # Reset the global index for this test
    if hasattr(FakeMultiExtractionModel, "_global_index"):
        FakeMultiExtractionModel._global_index = 0

    # Phase 1: Identification with more than max_objects
    objects = [
        {"name": f"Person{i}", "distinguishing_context": f"context {i}"}
        for i in range(15)  # Try to extract 15 objects
    ]

    identification_response = AIMessage(
        content="Identified many people",
        tool_calls=[
            {
                "id": f"identify_{uuid.uuid4()}",
                "name": "MultipleObjectIdentifiers",
                "args": {"objects": objects},
            }
        ],
    )

    # Phase 2: Create extraction responses for all identified objects
    # (but only max_objects should be processed)
    extraction_responses = []
    for i in range(15):
        extraction_responses.append(
            AIMessage(
                content=f"Extracted Person{i}",
                tool_calls=[
                    {
                        "id": f"person_{uuid.uuid4()}",
                        "name": "Person",
                        "args": {
                            "name": f"Person{i}",
                            "age": 20 + i,
                            "occupation": f"job{i}",
                        },
                    }
                ],
            )
        )

    model = FakeMultiExtractionModel(
        responses=[identification_response] + extraction_responses
    )

    max_limit = 10
    extractor = create_multi_object_extractor(
        model, target_schema=Person, max_objects=max_limit
    )

    result = await extractor.ainvoke("Many people are mentioned...")

    # Verify that max_objects limit is enforced
    assert len(result["responses"]) <= max_limit
    assert result["identification_count"] <= max_limit


@pytest.mark.asyncio
async def test_multi_object_extraction_empty_identification():
    """Test behavior when no objects are identified."""
    # Reset the global index for this test
    if hasattr(FakeMultiExtractionModel, "_global_index"):
        FakeMultiExtractionModel._global_index = 0

    # Phase 1: Identification with no objects
    identification_response = AIMessage(
        content="No objects found",
        tool_calls=[
            {
                "id": f"identify_{uuid.uuid4()}",
                "name": "MultipleObjectIdentifiers",
                "args": {"objects": []},
            }
        ],
    )

    model = FakeMultiExtractionModel(responses=[identification_response])

    extractor = create_multi_object_extractor(
        model, target_schema=Person, max_objects=5
    )

    result = await extractor.ainvoke("No people mentioned in this text.")

    # Verify results
    assert len(result["responses"]) == 0
    assert result["identification_count"] == 0
    assert result["attempts"] == 0


@pytest.mark.asyncio
async def test_multi_object_extraction_metadata():
    """Test that metadata includes object_index and stub information."""
    # Reset the global index for this test
    if hasattr(FakeMultiExtractionModel, "_global_index"):
        FakeMultiExtractionModel._global_index = 0

    # Phase 1: Identification response
    identification_response = AIMessage(
        content="Identified 2 people",
        tool_calls=[
            {
                "id": f"identify_{uuid.uuid4()}",
                "name": "MultipleObjectIdentifiers",
                "args": {
                    "objects": [
                        {"name": "Eve", "distinguishing_context": "doctor"},
                        {"name": "Frank", "distinguishing_context": "lawyer"},
                    ]
                },
            }
        ],
    )

    # Phase 2: Individual extraction responses
    eve_response = AIMessage(
        content="Extracted Eve",
        tool_calls=[
            {
                "id": f"person_{uuid.uuid4()}",
                "name": "Person",
                "args": {"name": "Eve", "age": 35, "occupation": "doctor"},
            }
        ],
    )

    frank_response = AIMessage(
        content="Extracted Frank",
        tool_calls=[
            {
                "id": f"person_{uuid.uuid4()}",
                "name": "Person",
                "args": {"name": "Frank", "age": 40, "occupation": "lawyer"},
            }
        ],
    )

    model = FakeMultiExtractionModel(
        responses=[identification_response, eve_response, frank_response]
    )

    extractor = create_multi_object_extractor(
        model, target_schema=Person, max_objects=5
    )

    result = await extractor.ainvoke(
        "Eve is a 35 year old doctor. Frank is a 40 year old lawyer."
    )

    # Verify metadata
    assert len(result["response_metadata"]) == 2

    # Check that object_index is present
    indices = [meta["object_index"] for meta in result["response_metadata"]]
    assert 0 in indices
    assert 1 in indices

    # Check that stub information is present
    for meta in result["response_metadata"]:
        assert "stub" in meta
        assert "name" in meta["stub"]
        assert "distinguishing_context" in meta["stub"]


@pytest.mark.asyncio
async def test_multi_object_extraction_nested_schema():
    """Test extraction with nested schema."""
    # Reset the global index for this test
    if hasattr(FakeMultiExtractionModel, "_global_index"):
        FakeMultiExtractionModel._global_index = 0

    # Phase 1: Identification
    identification_response = AIMessage(
        content="Identified users",
        tool_calls=[
            {
                "id": f"identify_{uuid.uuid4()}",
                "name": "MultipleObjectIdentifiers",
                "args": {
                    "objects": [
                        {"name": "User1", "distinguishing_context": "NYC user"},
                    ]
                },
            }
        ],
    )

    # Phase 2: Extraction
    user_response = AIMessage(
        content="Extracted User1",
        tool_calls=[
            {
                "id": f"person_{uuid.uuid4()}",
                "name": "UserWithAddress",
                "args": {
                    "name": "User1",
                    "address": {"city": "New York", "zipcode": "10001"},
                },
            }
        ],
    )

    model = FakeMultiExtractionModel(
        responses=[identification_response, user_response]
    )

    extractor = create_multi_object_extractor(
        model, target_schema=UserWithAddress
    )

    result = await extractor.ainvoke("User1 lives in NYC 10001")

    assert len(result["responses"]) == 1
    user = result["responses"][0]
    assert isinstance(user.address, Address)
    assert user.name == "User1"
    assert user.address.city == "New York"
    assert user.address.zipcode == "10001"