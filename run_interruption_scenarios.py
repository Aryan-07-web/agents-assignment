"""
Run interruption scenarios from the assignment prompt (no API keys required).

This uses the existing fake VAD/STT/LLM/TTS test harness to simulate:
1) Agent speaking + user says filler -> SHOULD NOT interrupt
2) Agent silent + user says "yeah" -> SHOULD be treated normally (agent proceeds)
3) Agent speaking + user says "no stop" -> SHOULD interrupt immediately
4) Agent speaking + user says "yeah okay but wait" -> SHOULD interrupt

How to run (PowerShell):
  cd agents-assignment
  python .\tests\run_interruption_scenarios.py

Take screenshots of the terminal output for proof.
"""

from __future__ import annotations

import asyncio
import os
import sys


# Make repo packages importable without installing them:
# - `livekit` package is in `livekit-agents/livekit/`
# - tests are in `tests/`
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LIVEKIT_AGENTS_PKG_ROOT = os.path.join(ROOT, "livekit-agents")
if LIVEKIT_AGENTS_PKG_ROOT not in sys.path:
    sys.path.insert(0, LIVEKIT_AGENTS_PKG_ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from livekit.agents.voice.agent import Agent  # noqa: E402
from livekit.agents.voice.agent_session import AgentSession  # noqa: E402
from livekit.agents.voice.events import PlaybackFinishedEvent  # noqa: E402
from livekit.agents.voice.transcription.synchronizer import (  # noqa: E402
    TranscriptSynchronizer,
    _SyncedAudioOutput,
)

from tests.fake_io import FakeAudioInput, FakeAudioOutput, FakeTextOutput  # noqa: E402
from tests.fake_llm import FakeLLM  # noqa: E402
from tests.fake_session import FakeActions  # noqa: E402
from tests.fake_stt import FakeSTT  # noqa: E402
from tests.fake_tts import FakeTTS  # noqa: E402
from tests.fake_vad import FakeVAD  # noqa: E402


SESSION_TIMEOUT = 60.0


class DemoAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful assistant.")


def _create_session(actions: FakeActions, *, speed_factor: float, extra_kwargs: dict | None = None) -> AgentSession:
    """
    Local copy of `tests.fake_session.create_session`, but importing AgentSession directly
    from `livekit.agents.voice.agent_session` to avoid `livekit.agents.__init__` (which imports
    optional CLI deps like numpy).
    """
    user_speeches = actions.get_user_speeches(speed_factor=speed_factor)
    llm_responses = actions.get_llm_responses(speed_factor=speed_factor)
    tts_responses = actions.get_tts_responses(speed_factor=speed_factor)

    stt = FakeSTT(fake_user_speeches=user_speeches)
    session = AgentSession[None](
        vad=FakeVAD(
            fake_user_speeches=user_speeches,
            min_silence_duration=0.5 / speed_factor,
            min_speech_duration=0.05 / speed_factor,
        ),
        stt=stt,
        llm=FakeLLM(fake_responses=llm_responses),
        tts=FakeTTS(fake_responses=tts_responses),
        min_interruption_duration=0.5 / speed_factor,
        min_endpointing_delay=0.5 / speed_factor,
        max_endpointing_delay=6.0 / speed_factor,
        false_interruption_timeout=2.0 / speed_factor,
        **(extra_kwargs or {}),
    )

    audio_input = FakeAudioInput()
    audio_output = FakeAudioOutput()
    transcription_output = FakeTextOutput()

    transcript_sync = TranscriptSynchronizer(
        next_in_chain_audio=audio_output,
        next_in_chain_text=transcription_output,
        speed=speed_factor,
    )
    session.input.audio = audio_input
    session.output.audio = transcript_sync.audio_output
    session.output.transcription = transcript_sync.text_output
    return session


async def _run_session(session: AgentSession, agent: Agent, *, drain_delay: float = 1.0) -> float:
    """
    Local copy of `tests.fake_session.run_session`, again avoiding `livekit.agents.__init__`.
    """
    import time
    import contextlib

    stt = session.stt
    audio_input = session.input.audio
    assert isinstance(stt, FakeSTT)
    assert isinstance(audio_input, FakeAudioInput)

    transcription_sync: TranscriptSynchronizer | None = None
    if isinstance(session.output.audio, _SyncedAudioOutput):
        transcription_sync = session.output.audio._synchronizer

    await session.start(agent)

    # start the fake vad and stt
    t_origin = time.time()
    audio_input.push(0.1)

    # wait for the user speeches to be processed
    await stt.fake_user_speeches_done

    await asyncio.sleep(drain_delay)
    with contextlib.suppress(RuntimeError):
        await session.drain()
    await session.aclose()

    if transcription_sync is not None:
        await transcription_sync.aclose()

    return t_origin


async def _run_one(
    *,
    title: str,
    actions: FakeActions,
    expect_interrupted: bool | None,
    speed: float = 5.0,
    extra_kwargs: dict | None = None,
) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)

    session = _create_session(actions, speed_factor=speed, extra_kwargs=extra_kwargs or {})
    playback_events: list[PlaybackFinishedEvent] = []
    session.output.audio.on("playback_finished", playback_events.append)

    await asyncio.wait_for(_run_session(session, DemoAgent()), timeout=SESSION_TIMEOUT)

    if not playback_events:
        print("playback_finished events: 0 (agent did not produce audio output)")
        if expect_interrupted is not None:
            print(f"EXPECTED interrupted={expect_interrupted}, but no playback was produced.")
        return

    # Usually 1 event for this harness
    ev = playback_events[-1]
    print(f"playback_finished: interrupted={ev.interrupted}, playback_position={ev.playback_position:.2f}s")

    if expect_interrupted is not None:
        assert ev.interrupted is expect_interrupted, (
            f"Expected interrupted={expect_interrupted}, got {ev.interrupted}"
        )
        print("[PASS] matched expected interruption behavior")


async def main() -> None:
    # Scenario 1: Long explanation, filler while agent speaking -> do NOT interrupt
    s1 = FakeActions()
    s1.add_user_speech(0.5, 2.5, "Tell me a long paragraph about history.")
    s1.add_llm("Here is a long paragraph about history...", duration=0.3)
    s1.add_tts(10.0)
    s1.add_user_speech(3.0, 4.0, "Okay yeah uh-huh", stt_delay=0.1)
    await _run_one(
        title="Scenario 1: Long explanation + filler while agent is speaking (should NOT interrupt)",
        actions=s1,
        expect_interrupted=False,
    )

    # Scenario 2: Agent asks and goes silent; user says "Yeah." -> treat normally (agent proceeds)
    # (We model this as a normal user turn that produces a short agent response.)
    s2 = FakeActions()
    s2.add_user_speech(0.5, 1.2, "Yeah.", stt_delay=0.1)
    s2.add_llm("Okay, starting now.", duration=0.2)
    s2.add_tts(2.0)
    await _run_one(
        title="Scenario 2: Agent silent + user says 'Yeah.' (should be processed normally)",
        actions=s2,
        expect_interrupted=False,
    )

    # Scenario 3: Correction "No stop." while agent speaking -> interrupt
    s3 = FakeActions()
    s3.add_user_speech(0.5, 2.5, "Count to ten.")
    s3.add_llm("One, two, three, four, five, six, seven, eight, nine, ten...", duration=0.3)
    s3.add_tts(10.0)
    s3.add_user_speech(3.0, 3.7, "No stop.", stt_delay=0.1)
    await _run_one(
        title="Scenario 3: Agent speaking + user says 'No stop.' (should interrupt)",
        actions=s3,
        expect_interrupted=True,
    )

    # Scenario 4: Mixed input "Yeah okay but wait." while agent speaking -> interrupt (contains 'wait')
    s4 = FakeActions()
    s4.add_user_speech(0.5, 2.5, "Explain photosynthesis.")
    s4.add_llm("Photosynthesis is the process by which plants...", duration=0.3)
    s4.add_tts(10.0)
    s4.add_user_speech(3.0, 4.0, "Yeah okay but wait.", stt_delay=0.1)
    await _run_one(
        title="Scenario 4: Agent speaking + user says 'Yeah okay but wait.' (should interrupt)",
        actions=s4,
        expect_interrupted=True,
    )

    print("\nAll scenarios completed.")


if __name__ == "__main__":
    asyncio.run(main())


