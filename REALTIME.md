# Realtime Voice AI Architecture

This document explains how LiveKit Agents achieve low-latency realtime voice interactions.

## Overview

LiveKit Agents supports two paths for voice AI:

| Path | Flow | Latency | Use Case |
|------|------|---------|----------|
| **Traditional Pipeline** | Audio → STT → LLM → TTS → Audio | Higher | Any LLM provider |
| **Realtime LLM** | Audio ↔ Realtime API (direct) | Lower | OpenAI Realtime, Gemini Live |

The realtime path bypasses STT/TTS processing by using models with native audio input/output modalities.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    LiveKit Room (WebRTC)                     │
│  Remote Participant Audio ← LiveKit Server → Local Track    │
└────────────┬────────────────────────────────────────────────┘
             │
             ↓
    ┌────────────────────┐
    │  RoomIO._input     │  WebRTC audio stream
    │  AudioInputStream  │
    └────────┬───────────┘
             │ rtc.AudioFrame
             ↓
    ┌────────────────────────────────────────┐
    │   AgentSession._forward_audio_task()   │
    └────────┬───────────────────────────────┘
             │
             ↓
    ┌────────────────────────────────────────┐
    │   AgentActivity.push_audio(frame)      │
    │                                        │
    │   ├─→ RealtimeSession.push_audio()     │  Direct to realtime API
    │   └─→ AudioRecognition.push_audio()    │  VAD + STT (fallback)
    └──────────┬─────────────────────────────┘
               │
               ↓
    ┌────────────────────────────────────────┐
    │   Realtime Model (e.g., OpenAI)        │
    │   - Server-side VAD                    │
    │   - Direct audio transcription         │
    │   - Native audio generation            │
    └────────┬───────────────────────────────┘
             │ GenerationCreatedEvent
             ↓
    ┌────────────────────────────────────────┐
    │   _realtime_generation_task()          │
    │                                        │
    │   Concurrent streams:                  │
    │   ├─ text_stream (transcript)          │
    │   ├─ audio_stream (voice output)       │
    │   └─ function_stream (tool calls)      │
    └────────┬───────────────────────────────┘
             │
             ↓
    ┌────────────────────────────────────────┐
    │  RoomIO._output                        │
    │  AudioSource → LocalAudioTrack         │
    └────────┬───────────────────────────────┘
             │ rtc.AudioFrame
             ↓
    ┌────────────────────────────────────────┐
    │ LiveKit Room (WebRTC)                  │
    │ → User hears response                  │
    └────────────────────────────────────────┘
```

## Key Components

### Core Interfaces

| File | Component | Purpose |
|------|-----------|---------|
| `llm/realtime.py` | `RealtimeModel` | Base class for realtime LLM providers |
| `llm/realtime.py` | `RealtimeSession` | Per-session audio/video streaming and generation |
| `llm/realtime.py` | `RealtimeCapabilities` | Declares model capabilities |
| `voice/agent_session.py` | `AgentSession` | Orchestrates I/O, models, and turn detection |
| `voice/agent_activity.py` | `AgentActivity` | Manages realtime generation lifecycle |
| `voice/room_io/` | `RoomIO` | WebRTC audio input/output via LiveKit |

### RealtimeCapabilities

Models declare their capabilities via `RealtimeCapabilities`:

```python
@dataclass
class RealtimeCapabilities:
    message_truncation: bool      # Can truncate messages mid-stream
    turn_detection: bool          # Server-side VAD support
    user_transcription: bool      # Built-in user speech transcription
    auto_tool_reply_generation: bool  # Auto-generates tool responses
    audio_output: bool            # Native audio output modality
    manual_function_calls: bool   # Manual function call control
```

### RealtimeSession Interface

The `RealtimeSession` abstract class defines the contract for realtime interactions:

```python
class RealtimeSession(ABC):
    # Audio/video streaming
    def push_audio(self, frame: rtc.AudioFrame) -> None: ...
    def push_video(self, frame: rtc.VideoFrame) -> None: ...

    # LLM context updates
    async def update_instructions(self, instructions: str) -> None: ...
    async def update_chat_ctx(self, chat_ctx: ChatContext) -> None: ...
    async def update_tools(self, tools: list[Tool]) -> None: ...

    # Generation control
    def generate_reply(self, *, instructions: str = ...) -> Future[GenerationCreatedEvent]: ...
    def interrupt(self) -> None: ...

    # Turn management (client-side)
    def commit_audio(self) -> None: ...
    def clear_audio(self) -> None: ...
    def truncate(self, *, message_id: str, ...) -> None: ...
```

### Events

Realtime sessions emit these events:

| Event | Description |
|-------|-------------|
| `input_speech_started` | Server-side VAD detected speech start |
| `input_speech_stopped` | Server-side VAD detected speech end |
| `input_audio_transcription_completed` | User speech transcribed |
| `generation_created` | New response generation started |
| `session_reconnected` | Session recovered after disconnect |
| `metrics_collected` | Performance metrics available |
| `error` | Error occurred |

### MessageGeneration

When a response is generated, it provides concurrent streams:

```python
@dataclass
class MessageGeneration:
    message_id: str
    text_stream: AsyncIterable[str]           # Streaming transcript
    audio_stream: AsyncIterable[rtc.AudioFrame]  # Streaming audio
    modalities: Awaitable[list[Literal["text", "audio"]]]
```

## Turn Detection Modes

The framework supports multiple turn detection strategies:

| Mode | Source | Description |
|------|--------|-------------|
| `realtime_llm` | Server VAD | Lowest latency, model handles turn detection |
| `vad` | Local VAD | Voice Activity Detector on client |
| `stt` | STT end-of-speech | Most reliable transcripts |
| `manual` | Explicit API | Custom turn control |

Priority fallback: `realtime_llm → vad → stt → manual`

## Usage Example

### Basic Realtime Agent

```python
from livekit import agents
from livekit.agents import AgentSession, Agent
from livekit.plugins import openai

async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    session = AgentSession(
        llm=openai.realtime.RealtimeModel(
            voice="coral",
            model="gpt-4o-realtime-preview"
        )
    )

    await session.start(
        room=ctx.room,
        agent=Agent(instructions="You are a helpful voice AI assistant.")
    )

    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
```

### With Tools

```python
from livekit.agents import function_tool

@function_tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"The weather in {location} is sunny, 72°F."

async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    session = AgentSession(
        llm=openai.realtime.RealtimeModel(voice="coral")
    )

    await session.start(
        room=ctx.room,
        agent=Agent(
            instructions="You are a helpful assistant with weather information.",
            tools=[get_weather]
        )
    )
```

## Tool Calling in Realtime Pipeline

The realtime pipeline handles tool/function calls differently from the traditional LLM pipeline, streaming function calls as they're generated rather than waiting for complete responses.

### Tool Calling Flow

```
┌─────────────────────────────────────────────────────────────┐
│  1. TOOL REGISTRATION                                        │
│                                                              │
│  Agent.tools=[get_weather, ...]                             │
│       ↓                                                      │
│  AgentActivity.update_tools()                               │
│       ↓                                                      │
│  RealtimeSession.update_tools()                             │
│       ↓                                                      │
│  OpenAI: session.update event with RealtimeFunctionTool[]   │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  2. FUNCTION CALL STREAMING                                  │
│                                                              │
│  User speaks → Model generates response                      │
│       ↓                                                      │
│  response.output_item.added (type: function_call)           │
│       ↓                                                      │
│  FunctionCall sent to function_ch channel                   │
│       ↓                                                      │
│  GenerationCreatedEvent.function_stream                     │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  3. CONCURRENT EXECUTION                                     │
│                                                              │
│  _execute_tools_task() loops:                               │
│       async for fnc_call in function_stream:                │
│           ├─ Lookup tool in ToolContext                     │
│           ├─ Validate & parse JSON arguments                │
│           └─ Execute tool concurrently (asyncio.create_task)│
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  4. RESULT FEEDBACK                                          │
│                                                              │
│  Tool results collected as FunctionCallOutput               │
│       ↓                                                      │
│  chat_ctx.items.extend(function_outputs)                    │
│       ↓                                                      │
│  RealtimeSession.update_chat_ctx()                          │
│       ↓                                                      │
│  OpenAI: conversation.item.create with function output      │
│       ↓                                                      │
│  Model continues generation with results                     │
└─────────────────────────────────────────────────────────────┘
```

### Key Data Structures

```python
# Function call from realtime API
@dataclass
class FunctionCall:
    call_id: str      # Unique identifier for this call
    name: str         # Tool function name
    arguments: str    # JSON string of arguments

# Result sent back to model
@dataclass
class FunctionCallOutput:
    call_id: str      # Matches the FunctionCall.call_id
    tool_use_id: str  # Provider-specific ID
    output: str       # Serialized result
    is_error: bool    # Whether execution failed

# Generation event with function stream
@dataclass
class GenerationCreatedEvent:
    message_stream: AsyncIterable[MessageGeneration]  # Text/audio
    function_stream: AsyncIterable[FunctionCall]      # Tool calls
    user_initiated: bool
    response_id: str | None
```

### Tool Execution Implementation

Location: `voice/generation.py`

```python
async def _execute_tools_task(
    *,
    function_stream: AsyncIterable[FunctionCall],
    tool_ctx: ToolContext,
    ...
) -> None:
    tasks: list[asyncio.Task] = []

    # Stream function calls as they arrive
    async for fnc_call in function_stream:
        # Lookup the function tool
        function_tool = tool_ctx.function_tools.get(fnc_call.name)
        if function_tool is None:
            logger.warning(f"unknown AI function `{fnc_call.name}`")
            continue

        # Parse JSON arguments
        fnc_args, fnc_kwargs = prepare_function_arguments(
            fnc=function_tool,
            json_arguments=fnc_call.arguments,
            call_ctx=RunContext(session=session, ...),
        )

        # Execute concurrently (don't block other function calls)
        task = asyncio.create_task(
            execute_tool(function_tool, fnc_args, fnc_kwargs)
        )
        tasks.append(task)

    # Wait for all tools to complete
    await asyncio.gather(*tasks)
```

### Realtime vs Traditional Tool Calling

| Aspect | Realtime Pipeline | Traditional Pipeline |
|--------|-------------------|----------------------|
| **Invocation** | Streamed via `function_stream` channel | Parsed from complete LLM response |
| **Registration** | `session.update` event | Request parameters or system prompt |
| **Execution** | Concurrent as functions stream in | Sequential after full response |
| **Result Feedback** | `update_chat_ctx()` → `conversation.item.create` | Included in next chat turn |
| **Streaming** | Real-time as model generates | Batch mode (wait for complete response) |
| **Interruption** | Can interrupt mid-function generation | Not applicable (already complete) |

### Capabilities Affecting Tools

```python
@dataclass
class RealtimeCapabilities:
    auto_tool_reply_generation: bool  # If True, model auto-replies after tool
    manual_function_calls: bool       # If True, agent controls tool invocation
```

For OpenAI Realtime:
- `auto_tool_reply_generation=False` - Agent must explicitly trigger reply
- `manual_function_calls=True` - Tools executed on agent side

### Key Methods

| Component | Method | Purpose |
|-----------|--------|---------|
| `AgentActivity` | `update_tools()` | Register tools with realtime session |
| `RealtimeSession` | `update_tools()` | Convert to provider format and send |
| `OpenAI Realtime` | `_create_tools_update_event()` | Build `session.update` event |
| `AgentActivity` | `_realtime_generation_task_impl()` | Orchestrate function handling |
| `generation.py` | `perform_tool_executions()` | Create execution task |
| `generation.py` | `_execute_tools_task()` | Main loop consuming function stream |

## Long-Running Tools and Async Execution

### Architectural Constraint: Speech Scheduling State Machine

The `AgentActivity` class uses a `_scheduling_paused` state machine to control when new speech can be scheduled. This has important implications for tools that take a long time to execute.

**Key behavior:**
1. When agent finishes speaking → `_scheduling_paused = True`
2. When user speaks (new turn) → `_scheduling_paused = False`
3. `session.say()` and `session.generate_reply()` check this flag before scheduling speech

**Location:** `voice/agent_activity.py:986-994`

```python
def _schedule_speech(self, speech: SpeechHandle, priority: int, force: bool = False) -> None:
    if self._scheduling_paused and not force:
        raise RuntimeError(
            "cannot schedule new speech, the speech scheduling is draining/pausing"
        )
```

### Why Background Task Injection Does NOT Work

A common desire is to have tools return immediately with an acknowledgment ("Searching...") while the actual work runs in a background asyncio task, then inject results when ready:

```python
# THIS DOES NOT WORK - DO NOT USE THIS PATTERN
@function_tool
async def slow_search(query: str) -> str:
    async def _background():
        result = await expensive_api_call(query)
        await session.say(f"Results: {result}")  # FAILS!

    asyncio.create_task(_background())
    return "Searching..."  # Returns immediately
```

**Why it fails:**
1. Tool returns "Searching...", agent speaks this
2. Agent finishes speaking → `_scheduling_paused = True`
3. User might ask another question, agent responds
4. Background task completes, calls `session.say()`
5. `_schedule_speech()` checks `_scheduling_paused`:
   - If True → `RuntimeError` or silent failure
   - Session tries `_next_activity` (for transitions) → usually `None` → fails

This is **by design** to prevent race conditions in the turn-based conversation model.

### Supported Pattern: Blocking with Interruption Support

The recommended approach for slow tools is to **block until completion** while allowing user interruption:

```python
from livekit.agents.voice.events import RunContext

@function_tool
async def slow_web_search(ctx: RunContext, query: str) -> str | None:
    """Search the web (takes ~3 seconds)."""

    async def _do_search():
        await asyncio.sleep(3)  # Simulate API delay
        return f"Results for '{query}': ..."

    # Start the background work
    search_task = asyncio.ensure_future(_do_search())

    # Wait for EITHER: task completion OR user interruption
    # Agent can still speak while we wait
    await ctx.speech_handle.wait_if_not_interrupted([search_task])

    if ctx.speech_handle.interrupted:
        # User interrupted - cancel and skip tool reply
        search_task.cancel()
        return None

    # Task completed - return result
    return search_task.result()
```

**How this works:**
1. User asks for slow search
2. Tool starts background coroutine but **blocks** waiting
3. Agent can speak while waiting (e.g., "Let me search for that...")
4. `wait_if_not_interrupted()` waits for task OR interruption
5. If user interrupts → task cancelled, returns `None` (skips tool reply)
6. If task completes → result returned, agent speaks it

**Reference implementation:** `examples/voice_agents/long_running_function.py`

### Disabling Interruptions

For tools where interruption would be problematic:

```python
@function_tool
async def critical_operation(ctx: RunContext, data: str) -> str:
    """Operation that must not be interrupted."""
    ctx.disallow_interruptions()  # Prevents user from interrupting

    result = await perform_critical_work(data)
    return result
```

### Key Limitation

With the blocking pattern, **the conversation is blocked** during tool execution. The user cannot ask other questions until the tool completes or is interrupted. This is a fundamental architectural constraint of the current realtime pipeline.

**What users CANNOT do:**
- Ask "slow search for AI news" → then ask "what time is it?" while search runs → then hear search results

**What users CAN do:**
- Ask "slow search for AI news" → wait 3 seconds → hear results
- Ask "slow search for AI news" → interrupt by speaking → search cancelled

### Summary: Tool Execution Patterns

| Pattern | Behavior | Use Case |
|---------|----------|----------|
| **Fast tool** | Return immediately | `get_time()`, `calculate()` |
| **Blocking + interruptible** | Block with `wait_if_not_interrupted()` | Web search, API calls |
| **Blocking + non-interruptible** | Block with `disallow_interruptions()` | Critical operations |
| **Background injection** | ❌ Not supported | - |

## Concurrent Streaming Architecture

This section explains how LiveKit Agents maintains continuous voice streaming while simultaneously processing events like tool calls, transcriptions, and interruptions.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CONCURRENT ASYNC TASKS                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────────┐   │
│   │  _send_task     │     │  _recv_task     │     │ _realtime_gen_task  │   │
│   │  (WebSocket TX) │     │  (WebSocket RX) │     │ (Event Processing)  │   │
│   └────────┬────────┘     └────────┬────────┘     └──────────┬──────────┘   │
│            │                       │                          │              │
│   ┌────────▼────────┐     ┌────────▼────────┐     ┌──────────▼──────────┐   │
│   │    _msg_ch      │     │   Async Iter    │     │  Parallel Streams   │   │
│   │  (Send Queue)   │     │  session.recv() │     │                     │   │
│   └────────┬────────┘     └────────┬────────┘     │  ├─ message_stream  │   │
│            │                       │              │  │  └─ audio_ch     │   │
│   Audio Frames ────►      ┌────────▼───────┐      │  │  └─ text_ch      │   │
│   Tool Responses──►       │ Route by Type  │      │  └─ function_stream │   │
│   Client Content─►        └───────┬────────┘      │     └─ function_ch  │   │
│                                   │               └──────────┬──────────┘   │
│                    ┌──────────────┼──────────────┐           │              │
│                    ▼              ▼              ▼           ▼              │
│            server_content    tool_call    usage_metadata   Tasks:           │
│                 │                │              │          - _read_messages │
│                 ▼                ▼              ▼          - _read_fnc_stream│
│            audio_ch.send   function_ch.send   emit()      - exe_task        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Design Patterns

**1. Separate Send/Receive Tasks**

Location: `plugins/google/realtime/realtime_api.py:727-731`

```python
send_task = asyncio.create_task(self._send_task(session), name="gemini-realtime-send")
recv_task = asyncio.create_task(self._recv_task(session), name="gemini-realtime-recv")
```

These run **concurrently** - audio can be sent while events are received. No blocking between input and output.

**2. Non-Blocking Channel for Outgoing Messages**

Location: `plugins/google/realtime/realtime_api.py:557-581`

```python
def push_audio(self, frame: rtc.AudioFrame) -> None:
    # Resample and chunk audio
    for nf in self._bstream.write(f.data.tobytes()):
        realtime_input = types.LiveClientRealtimeInput(media_chunks=[...])
        self._send_client_event(realtime_input)  # Non-blocking!

def _send_client_event(self, event: ClientEvents) -> None:
    with contextlib.suppress(utils.aio.channel.ChanClosed):
        self._msg_ch.send_nowait(event)  # Fire and forget
```

- `push_audio()` never blocks - it just queues frames
- `_send_task` drains the queue independently

**3. Async Channels for Response Streams**

Location: `plugins/google/realtime/realtime_api.py:87-95`

```python
@dataclass
class _ResponseGeneration:
    message_ch: utils.aio.Chan[llm.MessageGeneration]
    function_ch: utils.aio.Chan[llm.FunctionCall]  # Tool calls
    text_ch: utils.aio.Chan[str]                   # Transcription
    audio_ch: utils.aio.Chan[rtc.AudioFrame]       # Audio output
```

Each stream type has its own async channel. Consumers can read independently.

**4. Concurrent Stream Processing**

Location: `voice/agent_activity.py:2446-2489`

```python
# These run CONCURRENTLY:
tasks.append(asyncio.create_task(_read_messages(message_outputs)))  # Audio/text
tasks.append(asyncio.create_task(_read_fnc_stream()))               # Function calls
exe_task, tool_output = perform_tool_executions(function_stream=fnc_stream)

await speech_handle.wait_if_not_interrupted([*tasks])  # Wait for all
```

**5. Event Routing in Receive Task**

Location: `plugins/google/realtime/realtime_api.py:884-893`

```python
async for response in session.receive():
    if response.server_content:
        self._handle_server_content(response.server_content)  # Audio frames
    if response.tool_call:
        self._handle_tool_calls(response.tool_call)           # Function calls
    if response.tool_call_cancellation:
        self._handle_tool_call_cancellation(...)
    if response.usage_metadata:
        self._handle_usage_metadata(...)
```

Single receive loop dispatches to multiple handlers. Each handler pushes to its respective channel.

### Data Flow for Concurrent Operations

```
User speaks          Model generates           Tool executes
     │                     │                        │
     ▼                     ▼                        ▼
push_audio()         audio_ch.send()         function_ch.send()
     │                     │                        │
     ▼                     ▼                        ▼
 _msg_ch ──────►      _read_messages()       _read_fnc_stream()
     │                     │                        │
     ▼                     │                        │
_send_task()               │                        │
     │                     ▼                        ▼
     ▼               AudioOutput             perform_tool_executions()
  WebSocket              │                         │
     │                   ▼                         ▼
     ▼              Speaker plays           execute_tool() + emit event
  Gemini API
```

### Why Tool Execution Blocks (Despite Concurrent Streams)

The streams are concurrent, but **tool execution is awaited before turn completion**:

```python
# agent_activity.py:2489
await speech_handle.wait_if_not_interrupted([*tasks])

# Then later, after audio playout...
await exe_task  # This blocks waiting for tools!
```

This is a **design choice** by LiveKit Agents, not a technical limitation of the underlying SDK.

### Comparison: LiveKit Agents vs Raw Google GenAI SDK

The Google GenAI SDK itself does NOT block for tool execution:

```python
# Raw Google GenAI SDK - fully async, non-blocking
async for message in session.receive():
    if message.tool_call:
        # Fire and forget - no await!
        asyncio.create_task(background_search(session, message.tool_call))

    # Continue processing other messages immediately
    if message.server_content:
        play_audio(message.data)

async def background_search(session, tool_call):
    result = await slow_api_call()
    await session.send_tool_response(...)  # Injects result when ready
```

LiveKit Agents wraps this with orchestration that enforces sequential turn processing via the `_scheduling_paused` state machine.

### Summary: What's Concurrent vs What Blocks

| Component | Blocking? | Mechanism |
|-----------|-----------|-----------|
| Audio input (`push_audio`) | No | Queue to `_msg_ch` |
| Audio output (`server_content`) | No | Channel to `audio_ch` |
| Text transcription | No | Channel to `text_ch` |
| Function call events | No | Channel to `function_ch` |
| Tool execution | **Yes** | `await exe_task` before turn ends |
| Post-turn speech injection | **Blocked** | `_scheduling_paused` state |

The architecture is fully async and non-blocking for **streaming**, but enforces **sequential turn processing** for tool results.

## Realtime Plugins

### OpenAI Realtime

Location: `livekit-plugins/livekit-plugins-openai/livekit/plugins/openai/realtime/`

```python
from livekit.plugins import openai

model = openai.realtime.RealtimeModel(
    model="gpt-4o-realtime-preview",
    voice="coral",           # Voice selection
    modalities=["text", "audio"],  # Output modalities
    turn_detection=openai.realtime.RealtimeAudioInputTurnDetection(
        threshold=0.5,       # VAD sensitivity
        prefix_padding_ms=300,
        silence_duration_ms=500
    )
)
```

**Audio specs:** 24kHz, mono, PCM16

**WebSocket protocol:**
- `InputAudioBufferAppendEvent` - Push audio frames
- `InputAudioBufferCommitEvent` - Signal turn boundary
- `ResponseCreatedEvent` - Generation started
- `ResponseAudioDeltaEvent` - Audio chunks
- `ResponseTextDeltaEvent` - Transcript chunks

### Google Gemini Live

Location: `livekit-plugins/livekit-plugins-google/`

Similar interface with Gemini-specific options.

## Low-Latency Optimizations

1. **Direct audio streaming** - No HTTP round-trips, uses WebSocket/WebRTC
2. **Server-side VAD** - Eliminates client-side speech detection latency
3. **Native audio modality** - No STT/TTS conversion overhead
4. **Parallel processing** - Text, audio, and tool streams processed concurrently
5. **Preemptive generation** - Can start inference before confirmed turn end
6. **Optimized buffering** - 200ms queue with 50ms chunks for smooth playback

## Audio Pipeline Details

### Input Path

```python
class _ParticipantInputStream:
    def __init__(
        self,
        room: rtc.Room,
        track_source: rtc.TrackSource.ValueType,
        processor: rtc.FrameProcessor | None = None,  # Optional denoise
    ):
        self._data_ch = aio.Chan[T]()  # Async channel for frames
```

- Subscribes to `track_subscribed` events from LiveKit room
- Creates audio stream from remote participant track
- Supports dynamic participant switching

### Output Path

```python
class _ParticipantAudioOutput:
    def __init__(
        self,
        room: rtc.Room,
        sample_rate: int,      # e.g., 24000
        num_channels: int,     # e.g., 1 (mono)
    ):
        self._audio_source = rtc.AudioSource(
            sample_rate,
            num_channels,
            queue_size_ms=200  # Buffer for smooth playback
        )
```

- Chunks frames into 50ms buffers via `AudioByteStream`
- Publishes to LiveKit track with subscription confirmation
- Supports pause/resume for false interruption handling

## Comparison: Traditional vs Realtime

| Aspect | Traditional Pipeline | Realtime LLM |
|--------|---------------------|--------------|
| Latency | 500ms-2s | <100ms |
| Components | STT + LLM + TTS | Single API |
| Turn detection | Client-side VAD/STT | Server-side VAD |
| Provider support | Any LLM | OpenAI, Gemini |
| Cost | Per-component pricing | Unified pricing |
| Customization | Mix providers | Single provider |

## Key Source Files

| File | Lines | Purpose |
|------|-------|---------|
| `llm/realtime.py` | ~210 | Core realtime interface |
| `voice/agent_session.py` | ~1,325 | Session orchestration |
| `voice/agent_activity.py` | ~2,790 | Realtime generation lifecycle |
| `voice/generation.py` | ~700 | Tool execution and audio generation |
| `voice/audio_recognition.py` | ~500 | Turn detection & speech recognition |
| `voice/room_io/_input.py` | ~200 | WebRTC audio input |
| `voice/room_io/_output.py` | ~200 | WebRTC audio output |
| `plugins-openai/realtime/realtime_model.py` | ~1,000 | OpenAI implementation |

## See Also

- [LiveKit Agents Documentation](https://docs.livekit.io/agents/)
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime)
- [CLAUDE.md](./CLAUDE.md) - Development commands and architecture overview
