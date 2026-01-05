"""Example test demonstrating multi-object extraction with Spirit Airlines aircraft data.

This module shows how to use create_multi_object_extractor with real-world aviation data
from Spirit Airlines aircraft specification sheets. These examples can serve as templates
for extracting structured data from PDF documents.

Note: These are example tests showing the expected usage pattern. For actual extraction,
you would replace the mock LLM with a real model like ChatOpenAI.
"""

import sys
from pathlib import Path

from pydantic import BaseModel, Field

# Import the sample Pydantic models
sys.path.insert(0, str(Path(__file__).parent.parent / "sample"))
from sample_pydantic_model import (  # noqa: E402
    APUInfo,
    EngineInfo,
    LandingGear,
    MaintenanceCheck,
)


def test_engine_schema_structure():
    """Test that the EngineInfo schema correctly models aircraft engine data."""
    engine = EngineInfo(
        position="1",
        esn="V18452",
        engine_type="V2533-A5",
        tsn=21021,
        csn=9252,
        tso=21021,
        cso=9252,
        limiter_cycles=10748,
        limiter_part="VARIOUS limiting parts",
    )

    assert engine.position == "1"
    assert engine.esn == "V18452"
    assert engine.tsn == 21021
    assert engine.limiter_cycles == 10748


def test_maintenance_check_schema_structure():
    """Test that the MaintenanceCheck schema correctly models maintenance data."""
    check = MaintenanceCheck(
        check_type="C-Check",
        last_completed="3/22/2023",
        next_due="3/22/2026",
        remarks="36 months interval",
    )

    assert check.check_type == "C-Check"
    assert check.last_completed == "3/22/2023"
    assert check.next_due == "3/22/2026"


def test_landing_gear_schema_structure():
    """Test that the LandingGear schema correctly models landing gear data."""
    gear = LandingGear(
        position="Nose",
        last_overhaul=None,
        next_due="3/31/2029",
        tsn=25003,
        csn=11083,
    )

    assert gear.position == "Nose"
    assert gear.tsn == 25003
    assert gear.next_due == "3/31/2029"


def test_apu_schema_structure():
    """Test that the APUInfo schema correctly models APU data."""
    apu = APUInfo(
        manufacturer="Honeywell",
        model="GTCP-131-9A",
        serial_number="P-10563",
        status="in service",
        tso=2299,
        cso=3353,
        tsn=2299,
        csn=3353,
    )

    assert apu.manufacturer == "Honeywell"
    assert apu.model == "GTCP-131-9A"
    assert apu.tsn == 2299


def test_example_extraction_pattern():
    """
    Example showing how to extract multiple engines from Spirit Airlines spec data.

    This demonstrates the intended usage pattern with create_multi_object_extractor.
    In production, you would use a real LLM instead of the mock data shown here.

    Example usage:
    ```python
    from langchain_openai import ChatOpenAI
    from trustcall import create_multi_object_extractor
    from sample_pydantic_model import EngineInfo

    llm = ChatOpenAI(model="gpt-4o")
    extractor = create_multi_object_extractor(
        llm,
        target_schema=EngineInfo,
        max_objects=4  # A321 has 2 engines
    )

    # Extract from Spirit Airlines spec text
    spirit_spec_text = '''
    Aircraft Type: A321-231
    Engine Type: V2533-A5

    Engines:
    Position 1: ESN V18452, TSN 21,021.05 hours, CSN 9,252 cycles
    Position 2: ESN V18453, TSN 25,253.73 hours, CSN 11,160 cycles

    Engine 1 has 10,748 cycles remaining until limiter.
    Engine 2 has 8,840 cycles remaining until limiter.
    '''

    result = extractor.invoke(spirit_spec_text)

    # Access extracted engines
    for engine in result["responses"]:
        print(f"Engine {engine.position}: ESN {engine.esn}, "
              f"TSN {engine.tsn}, Remaining cycles: {engine.limiter_cycles}")
    ```
    """
    # This is a documentation test - the actual extraction would happen with a real LLM
    assert True, "See docstring for extraction pattern example"


def test_example_multi_check_extraction_pattern():
    """
    Example showing how to extract multiple maintenance checks.

    Example usage:
    ```python
    from langchain_openai import ChatOpenAI
    from trustcall import create_multi_object_extractor
    from sample_pydantic_model import MaintenanceCheck

    llm = ChatOpenAI(model="gpt-4o")
    extractor = create_multi_object_extractor(
        llm,
        target_schema=MaintenanceCheck,
        max_objects=10
    )

    maintenance_text = '''
    Maintenance Inspection Data:

    C-Check: Last completed 3/22/2023, Next due 3/22/2026 (36 months interval)
    6-Year Check: Last completed 3/22/2023, Next due 3/22/2029 (72 months interval)
    12-Year Check: Next due 3/22/2029 (144 months interval)
    '''

    result = extractor.invoke(maintenance_text)

    # Access extracted maintenance checks
    for check in result["responses"]:
        print(f"{check.check_type}: Last {check.last_completed}, "
              f"Next due {check.next_due}")
    ```
    """
    # This is a documentation test - the actual extraction would happen with a real LLM
    assert True, "See docstring for extraction pattern example"
