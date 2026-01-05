# Multi-Object Extraction Feature

## Overview

The multi-object extraction feature adds a new `create_multi_object_extractor()` function that uses a two-phase approach to extract multiple objects of the same pydantic model from a single context.

## Problem Solved

When extracting multiple objects of the same schema from a single context (e.g., extracting 5 `Person` objects from a conversation), LLMs can struggle with:
- Inconsistent quality across objects
- Missing objects or incomplete extraction
- Higher token costs due to large output generation

## Solution

The `create_multi_object_extractor()` uses a two-phase extraction approach with LangGraph's `Send` API for parallel processing:

### Phase 1: Identification
- Single LLM call identifies basic information for each object
- Returns a list of "stubs" with minimal identifying information
- Does NOT use validation/retry logic - it's a simple identification pass

### Phase 2: Parallel Enrichment
- Uses LangGraph's `Send` API to fan-out to parallel nodes
- Each node extracts a SINGLE pydantic model object
- Each parallel node performs validation on the extracted object
- Results are aggregated using `operator.add` reducer for fan-in

## API

```python
from trustcall import create_multi_object_extractor
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str = Field(description="Person's full name")
    age: int = Field(description="Person's age")
    occupation: str = Field(description="Person's occupation")

# Create extractor
extractor = create_multi_object_extractor(
    llm="gpt-4",  # or any BaseChatModel instance
    target_schema=Person,
    max_objects=10,  # safety limit
)

# Extract multiple people
result = extractor.invoke(
    "Alice is 30 and works as an engineer. Bob is 25 and is a designer."
)

# Result contains:
# - responses: List of extracted Person objects
# - identification_count: Number of objects identified
# - attempts: Total attempts across all extractions
# - response_metadata: Metadata for each object
```

## Key Features

1. **Configurable Identification Schema**: Customize Phase 1 identification fields
2. **Safety Limits**: `max_objects` parameter prevents runaway extractions
3. **Parallel Processing**: Efficient extraction using LangGraph's Send API
4. **Metadata Tracking**: Each extracted object includes index and stub information
5. **Type Safety**: Full pydantic validation for extracted objects

## Testing

Comprehensive test suite with 6 tests covering:
- Basic multi-object extraction
- Validation during extraction
- Custom identification schemas
- Max objects limit enforcement
- Empty identification handling
- Metadata tracking

All tests pass successfully.

## Files Modified

- `trustcall/_base.py`: Added multi-object extraction implementation
- `trustcall/__init__.py`: Exported new function and types
- `tests/unit_tests/test_multi_object_extraction.py`: Comprehensive test suite

## Future Enhancements

- Full validation/retry with patches in parallel nodes (currently simplified)
- Support for heterogeneous object extraction
- Streaming support for large-scale extractions
- Advanced error handling and recovery strategies
