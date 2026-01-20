"""
Tests for intelligent interruption handling.

These tests verify that the agent distinguishes between:
- Passive acknowledgements (ignore while agent is speaking)
- Active interruptions (stop immediately)
"""

from __future__ import annotations

import asyncio

import pytest

from livekit.agents import (
    Agent,
    AgentStateChangedEvent,
)
from livekit.agents.voice.events import PlaybackFinishedEvent

from .fake_session import FakeActions, create_session, run_session

SESSION_TIMEOUT = 60.0


class TestAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful assistant.")


def check_timestamp(actual: float, expected: float, *, speed_factor: float = 1.0, tolerance: float = 0.2) -> None:
    """Check if timestamp is within tolerance."""
    expected_scaled = expected / speed_factor
    assert abs(actual - expected_scaled) < tolerance / speed_factor, (
        f"timestamp mismatch: expected {expected_scaled:.2f}, got {actual:.2f}"
    )


@pytest.mark.asyncio
async def test_ignore_filler_words_while_agent_speaking() -> None:
    """
    Test that filler words like "yeah", "ok" don't interrupt agent when speaking.
    """
    speed = 5.0
    actions = FakeActions()
    # User asks a question
    actions.add_user_speech(0.5, 2.5, "Tell me about the weather.")
    # Agent starts responding
    actions.add_llm("The weather today is sunny and warm.", duration=0.3)
    actions.add_tts(10.0)  # Long response that will be playing
    
    # User says filler words while agent is speaking (at 3.0s, during playback)
    actions.add_user_speech(3.0, 3.5, "yeah", stt_delay=0.1)
    actions.add_user_speech(4.0, 4.5, "ok", stt_delay=0.1)
    actions.add_user_speech(5.0, 5.5, "hmm", stt_delay=0.1)
    
    # Use default ignore words
    session = create_session(actions, speed_factor=speed)
    agent = TestAgent()
    
    agent_state_events: list[AgentStateChangedEvent] = []
    playback_finished_events: list[PlaybackFinishedEvent] = []
    session.on("agent_state_changed", agent_state_events.append)
    session.output.audio.on("playback_finished", playback_finished_events.append)
    
    t_origin = await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)
    
    # Agent should continue speaking without interruption
    # Find when agent started speaking
    speaking_started = None
    for event in agent_state_events:
        if event.new_state == "speaking":
            speaking_started = event.created_at - t_origin
    
    assert speaking_started is not None, "Agent should have started speaking"
    
    # Check that agent wasn't interrupted by filler words
    # The playback should finish normally (not interrupted)
    assert len(playback_finished_events) == 1
    assert playback_finished_events[0].interrupted is False, (
        "Agent should not be interrupted by filler words"
    )


@pytest.mark.asyncio
async def test_interrupt_with_stop_word() -> None:
    """
    Test that interrupt words like "stop" interrupt agent immediately.
    """
    speed = 5.0
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Tell me a long story.")
    actions.add_llm("Once upon a time, there was a very long story...", duration=0.3)
    actions.add_tts(10.0)  # Long response
    
    # User says "stop" while agent is speaking (at 3.0s)
    actions.add_user_speech(3.0, 3.5, "stop", stt_delay=0.1)
    
    session = create_session(actions, speed_factor=speed)
    agent = TestAgent()
    
    agent_state_events: list[AgentStateChangedEvent] = []
    playback_finished_events: list[PlaybackFinishedEvent] = []
    session.on("agent_state_changed", agent_state_events.append)
    session.output.audio.on("playback_finished", playback_finished_events.append)
    
    t_origin = await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)
    
    # Agent should be interrupted
    assert len(playback_finished_events) == 1
    assert playback_finished_events[0].interrupted is True, (
        "Agent should be interrupted by 'stop' word"
    )
    
    # Check agent state changed to listening after interruption
    listening_after_interrupt = False
    for i, event in enumerate(agent_state_events):
        if i > 0 and event.new_state == "listening":
            # Check if this happened after speaking started
            prev_event = agent_state_events[i - 1]
            if prev_event.new_state == "speaking":
                listening_after_interrupt = True
                break
    
    assert listening_after_interrupt, "Agent should transition to listening after interruption"


@pytest.mark.asyncio
async def test_interrupt_with_wait_word() -> None:
    """
    Test that "wait" interrupts agent even when mixed with filler words.
    """
    speed = 5.0
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Explain quantum physics.")
    actions.add_llm("Quantum physics is a complex topic...", duration=0.3)
    actions.add_tts(10.0)
    
    # User says "yeah okay but wait" - should interrupt because of "wait"
    actions.add_user_speech(3.0, 4.0, "yeah okay but wait", stt_delay=0.1)
    
    session = create_session(actions, speed_factor=speed)
    agent = TestAgent()
    
    playback_finished_events: list[PlaybackFinishedEvent] = []
    session.output.audio.on("playback_finished", playback_finished_events.append)
    
    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)
    
    # Should interrupt because "wait" is an interrupt word
    assert len(playback_finished_events) == 1
    assert playback_finished_events[0].interrupted is True, (
        "Agent should be interrupted by 'wait' even when mixed with filler words"
    )


@pytest.mark.asyncio
async def test_normal_behavior_when_agent_silent() -> None:
    """
    Test that when agent is silent, all user input (including filler words) is treated normally.
    """
    speed = 5.0
    actions = FakeActions()
    # User says filler words while agent is silent
    actions.add_user_speech(0.5, 1.5, "yeah")
    actions.add_user_speech(2.0, 3.0, "ok")
    
    session = create_session(actions, speed_factor=speed)
    agent = TestAgent()
    
    agent_state_events: list[AgentStateChangedEvent] = []
    session.on("agent_state_changed", agent_state_events.append)
    
    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)
    
    # Agent should process the input normally (even though they're filler words)
    # The agent should transition to thinking/listening states
    assert len(agent_state_events) >= 2
    # Agent should be in listening state (not stuck)
    final_state = agent_state_events[-1].new_state
    assert final_state in ("listening", "thinking"), (
        "Agent should process input normally when silent, even if it's filler words"
    )


@pytest.mark.asyncio
async def test_custom_ignore_words() -> None:
    """
    Test that custom ignore words can be configured.
    """
    speed = 5.0
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Tell me something.")
    actions.add_llm("Here is some information for you.", duration=0.3)
    actions.add_tts(10.0)
    
    # User says custom ignore word
    actions.add_user_speech(3.0, 3.5, "sure", stt_delay=0.1)
    
    # Configure custom ignore words
    session = create_session(
        actions,
        speed_factor=speed,
        extra_kwargs={"interruption_ignore_words": ["sure", "right", "got it"]}
    )
    agent = TestAgent()
    
    playback_finished_events: list[PlaybackFinishedEvent] = []
    session.output.audio.on("playback_finished", playback_finished_events.append)
    
    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)
    
    # Should not interrupt because "sure" is in ignore list
    assert len(playback_finished_events) == 1
    assert playback_finished_events[0].interrupted is False, (
        "Agent should not be interrupted by custom ignore word 'sure'"
    )


@pytest.mark.asyncio
async def test_custom_interrupt_words() -> None:
    """
    Test that custom interrupt words can be configured.
    """
    speed = 5.0
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Tell me a story.")
    actions.add_llm("Once upon a time...", duration=0.3)
    actions.add_tts(10.0)
    
    # User says custom interrupt word
    actions.add_user_speech(3.0, 3.5, "halt", stt_delay=0.1)
    
    # Configure custom interrupt words
    session = create_session(
        actions,
        speed_factor=speed,
        extra_kwargs={"interruption_interrupt_words": ["halt", "cease", "abort"]}
    )
    agent = TestAgent()
    
    playback_finished_events: list[PlaybackFinishedEvent] = []
    session.output.audio.on("playback_finished", playback_finished_events.append)
    
    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)
    
    # Should interrupt because "halt" is in interrupt list
    assert len(playback_finished_events) == 1
    assert playback_finished_events[0].interrupted is True, (
        "Agent should be interrupted by custom interrupt word 'halt'"
    )


@pytest.mark.asyncio
async def test_multiple_filler_words_ignored() -> None:
    """
    Test that multiple filler words in sequence are all ignored.
    """
    speed = 5.0
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "What's the weather?")
    actions.add_llm("The weather is nice today.", duration=0.3)
    actions.add_tts(10.0)
    
    # User says multiple filler words
    actions.add_user_speech(3.0, 3.5, "yeah", stt_delay=0.1)
    actions.add_user_speech(4.0, 4.5, "ok", stt_delay=0.1)
    actions.add_user_speech(5.0, 5.5, "hmm", stt_delay=0.1)
    actions.add_user_speech(6.0, 6.5, "right", stt_delay=0.1)
    
    session = create_session(actions, speed_factor=speed)
    agent = TestAgent()
    
    playback_finished_events: list[PlaybackFinishedEvent] = []
    session.output.audio.on("playback_finished", playback_finished_events.append)
    
    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)
    
    # Should not interrupt - all are filler words
    assert len(playback_finished_events) == 1
    assert playback_finished_events[0].interrupted is False, (
        "Agent should not be interrupted by multiple filler words"
    )


@pytest.mark.asyncio
async def test_filler_words_followed_by_real_input() -> None:
    """
    Test that filler words followed by real input should interrupt.
    """
    speed = 5.0
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Tell me about dogs.")
    actions.add_llm("Dogs are wonderful pets...", duration=0.3)
    actions.add_tts(10.0)
    
    # User says filler words then real question
    actions.add_user_speech(3.0, 4.5, "yeah ok what about cats", stt_delay=0.1)
    
    session = create_session(actions, speed_factor=speed)
    agent = TestAgent()
    
    playback_finished_events: list[PlaybackFinishedEvent] = []
    session.output.audio.on("playback_finished", playback_finished_events.append)
    
    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)
    
    # Should interrupt because "what about cats" contains non-ignore words
    assert len(playback_finished_events) == 1
    assert playback_finished_events[0].interrupted is True, (
        "Agent should be interrupted when filler words are followed by real input"
    )


@pytest.mark.asyncio
async def test_interrupt_word_in_middle_of_sentence() -> None:
    """
    Test that interrupt words work even when in the middle of a sentence.
    """
    speed = 5.0
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Explain something.")
    actions.add_llm("Let me explain...", duration=0.3)
    actions.add_tts(10.0)
    
    # User says sentence with interrupt word in middle
    actions.add_user_speech(3.0, 4.5, "I need you to stop now", stt_delay=0.1)
    
    session = create_session(actions, speed_factor=speed)
    agent = TestAgent()
    
    playback_finished_events: list[PlaybackFinishedEvent] = []
    session.output.audio.on("playback_finished", playback_finished_events.append)
    
    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)
    
    # Should interrupt because "stop" is in the sentence
    assert len(playback_finished_events) == 1
    assert playback_finished_events[0].interrupted is True, (
        "Agent should be interrupted when interrupt word appears in sentence"
    )

