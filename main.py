import asyncio
import sys
import json
import threading
from pathlib import Path
import requests
from videosdk.agents import Agent, AgentSession, RealTimePipeline, JobContext, RoomOptions, WorkerJob, MCPServerStdio
from videosdk.agents.job import get_current_job_context
from videosdk.utils import PubSubPublishConfig
from videosdk.plugins.google import GeminiRealtime, GeminiLiveConfig
from google.genai.types import AudioTranscriptionConfig
from videosdk.plugins.simli import SimliAvatar, SimliConfig
from dotenv import load_dotenv
import os

load_dotenv(override=True)

# Default interview instructions (used as fallback)
DEFAULT_INSTRUCTIONS = (
    "You are Aiva, an AI technical interviewer conducting a technical interview session in English. "
    "Your objective is to ask the interviewee questions strictly from the predefined set of question_bank. "
    "REMEMBER: 1. You prompt interviewee if responses go against ethical interview conduction guidelines. "
    "2. DO NOT answer the question yourself in any scenario. 3. Do not evaluate the answer yourself. "
    "4. If question is to be rephrased, always use exact question text from question_bank without any creativity. "
    "5. Present the coding question text exactly as stored in the question_bank. "
    "6. Keep progressing with the interview with unique questions without being repetitive."
)

def get_room_id(auth_token: str) -> str:
    url = "https://api.videosdk.live/v2/rooms"
    headers = {"Authorization": auth_token}
    response = requests.post(url, headers=headers)
    response.raise_for_status()
    return response.json()["roomId"]


def fetch_interview_config(interview_id: str) -> dict:
    """Fetch interview-specific configuration from the backend API."""
    backend_url = os.getenv("BACKEND_API_URL", "http://localhost:8084/api/v1")
    try:
        response = requests.get(
            f"{backend_url}/interview/{interview_id}",
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[Agent] Failed to fetch interview config for {interview_id}: {e}")
        return {}


class InterviewAgent(Agent):
    """AI Interview Agent that uses dynamic instructions fetched from backend."""

    def __init__(self, instructions: str = DEFAULT_INSTRUCTIONS, candidate_name: str = ""):
        mcp_script_weather = Path(__file__).parent / "mcp_weather.py"
        super().__init__(
            instructions=instructions,
            mcp_servers=[
                MCPServerStdio(
                    executable_path=sys.executable,
                    process_arguments=[str(mcp_script_weather)],
                    session_timeout=30
                )
            ]
        )
        self.candidate_name = candidate_name

    async def _publish_chat(self, role: str, content: str):
        """Publish a chat message to the CHAT_MESSAGE PubSub topic."""
        try:
            ctx = get_current_job_context()
            if ctx and ctx.room:
                msg = json.dumps({"role": role, "content": content})
                await ctx.room.publish_to_pubsub(
                    PubSubPublishConfig(
                        topic="CHAT_MESSAGE",
                        message=msg,
                        options={"persist": True}
                    )
                )
        except Exception as e:
            print(f"[Agent] PubSub publish error: {e}", flush=True)

    async def _publish_status(self, status: str):
        """Publish agent status (speaking/listening/idle) to AGENT_STATUS topic."""
        try:
            ctx = get_current_job_context()
            if ctx and ctx.room:
                msg = json.dumps({"status": status})
                await ctx.room.publish_to_pubsub(
                    PubSubPublishConfig(topic="AGENT_STATUS", message=msg)
                )
        except Exception as e:
            print(f"[Agent] PubSub status error: {e}", flush=True)

    async def on_enter(self) -> None:
        greeting = f"Hello{' ' + self.candidate_name if self.candidate_name else ''}! I'm Aiva, your AI interviewer. Let's begin the interview."
        print(f"[Agent] on_enter: greeting immediately", flush=True)
        # Publish greeting text immediately so user sees it without waiting for audio
        await self._publish_status("speaking")
        await self._publish_chat("assistant", greeting)
        try:
            await self.session.say(greeting)
        except Exception as e:
            print(f"[Agent] on_enter: session.say() FAILED: {e}", flush=True)
        await self._publish_status("listening")
        print(f"[Agent] on_enter: done, now listening", flush=True)

    async def on_exit(self) -> None:
        farewell = "Thank you for participating in the interview! Goodbye."
        await self._publish_status("speaking")
        await self._publish_chat("assistant", farewell)
        await self.session.say(farewell)
        await self._publish_status("idle")

    def on_speech_in(self, data: dict) -> None:
        print(f"[Agent] on_speech_in: received user audio", flush=True)

    def on_speech_out(self, data: dict) -> None:
        print(f"[Agent] on_speech_out: agent producing audio", flush=True)


async def start_session(context: JobContext):
    """Main entry point for agent session."""
    # Initialize Gemini Realtime model with explicit English language
    model = GeminiRealtime(
        model="gemini-2.5-flash-native-audio-latest",
        api_key=os.getenv("GOOGLE_API_KEY"),
        config=GeminiLiveConfig(
            voice="Leda",
            language_code="en-US",
            response_modalities=["AUDIO"],
            input_audio_transcription=AudioTranscriptionConfig(),
            output_audio_transcription=AudioTranscriptionConfig()
        )
    )

    # Try Simli avatar with LIVEKIT transport; fall back to audio-only if it fails
    avatar = None
    simli_api_key = os.getenv("SIMLI_API_KEY")
    simli_face_id = os.getenv("SIMLI_FACE_ID")
    if simli_api_key and simli_face_id:
        try:
            avatar = SimliAvatar(
                config=SimliConfig(face_id=simli_face_id),
                api_key=simli_api_key,
                transport_mode="LIVEKIT"
            )
            print("[Agent] Simli avatar configured (LIVEKIT mode)", flush=True)
        except Exception as e:
            print(f"[Agent] Simli config failed, using audio-only: {e}", flush=True)
            avatar = None
    else:
        print("[Agent] No Simli credentials, using audio-only mode", flush=True)

    pipeline = RealTimePipeline(model=model, avatar=avatar) if avatar else RealTimePipeline(model=model)

    # Get instructions from context metadata (set by HTTP API) or use default
    instructions = getattr(context, '_interview_instructions', DEFAULT_INSTRUCTIONS)
    candidate_name = getattr(context, '_candidate_name', '')

    agent = InterviewAgent(instructions=instructions, candidate_name=candidate_name)
    session = AgentSession(
        agent=agent,
        pipeline=pipeline
    )

    # Publish agent state changes to AGENT_STATUS PubSub topic
    def on_agent_state(data):
        try:
            state = data.get("state", "") if isinstance(data, dict) else str(data)
            if state == "speaking":
                asyncio.create_task(agent._publish_status("speaking"))
            elif state in ("listening", "idle"):
                asyncio.create_task(agent._publish_status("listening"))
        except Exception:
            pass

    session.on("agent_state_changed", on_agent_state)

    # Transcription handler - publishes to CHAT_MESSAGE PubSub
    # Sends both interim (is_final=false) and final transcriptions for low-latency display
    def on_transcription(data):
        try:
            role = data.get("role", "")
            text = data.get("text", "")
            is_final = data.get("is_final", False)
            if text.strip():
                pubsub_role = "user" if role == "user" else "assistant"
                if is_final:
                    safe_text = text[:80].encode('ascii', 'replace').decode('ascii')
                    print(f"[Agent] Transcription ({pubsub_role}): {safe_text}", flush=True)
                # Publish with is_final flag so frontend can show interim text immediately
                msg = json.dumps({"role": pubsub_role, "content": text.strip(), "is_final": is_final})
                async def _pub():
                    try:
                        ctx = get_current_job_context()
                        if ctx and ctx.room:
                            await ctx.room.publish_to_pubsub(
                                PubSubPublishConfig(
                                    topic="CHAT_MESSAGE",
                                    message=msg,
                                    options={"persist": is_final}
                                )
                            )
                    except Exception as e:
                        print(f"[Agent] PubSub publish error: {e}", flush=True)
                asyncio.create_task(_pub())
        except Exception as e:
            safe_err = str(e).encode('ascii', 'replace').decode('ascii')
            print(f"[Agent] Transcription handler error: {safe_err}", flush=True)

    pipeline.on("realtime_model_transcription", on_transcription)

    try:
        print("[Agent] Connecting to VideoSDK room...", flush=True)
        await context.connect()
        print("[Agent] Connected! Starting session...", flush=True)
        await session.start()
        print("[Agent] Session started, pipeline active. Waiting for interactions...", flush=True)
        await asyncio.Event().wait()
    except Exception as e:
        print(f"[Agent] FATAL ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        await session.close()
        await context.shutdown()


def make_context(room_id: str = None, auth_token: str = None,
                 interview_id: str = None) -> JobContext:
    """Create a JobContext for the agent session.

    Can be called with explicit parameters (from HTTP API) or use env vars (standalone mode).
    """
    if not auth_token:
        auth_token = os.getenv("VIDEOSDK_AUTH_TOKEN")
    if not room_id:
        room_id = get_room_id(auth_token)

    playground_url = f"https://playground.videosdk.live?token={auth_token}&meetingId={room_id}"
    print(f"\nPlayground URL:\n{playground_url}\n", flush=True)

    room_options = RoomOptions(
        room_id=room_id,
        auth_token=auth_token,
        name="Aiva Interview Agent",
        playground=True
    )
    context = JobContext(room_options=room_options)

    # If interview_id provided, fetch config from backend and attach to context
    if interview_id:
        config = fetch_interview_config(interview_id)
        if config:
            # Build instructions from interview config
            instructions = config.get('instructions', DEFAULT_INSTRUCTIONS)
            context._interview_instructions = instructions
            context._candidate_name = config.get('candidate', '')
            print(f"[Agent] Loaded interview config for interview {interview_id}")
        else:
            context._interview_instructions = DEFAULT_INSTRUCTIONS
            context._candidate_name = ''
    else:
        context._interview_instructions = DEFAULT_INSTRUCTIONS
        context._candidate_name = ''

    return context


# ============================================================
# HTTP API Server (FastAPI) for receiving session requests
# from the Next.js frontend
# ============================================================

def run_http_server():
    """Start a FastAPI HTTP server for receiving session requests."""
    try:
        from fastapi import FastAPI, BackgroundTasks
        from pydantic import BaseModel
        import uvicorn
    except ImportError:
        print("[Agent] FastAPI/uvicorn not installed. Running in standalone mode only.")
        print("[Agent] Install with: pip install fastapi uvicorn")
        return

    app = FastAPI(title="Aiva Interview Agent Service")

    # Track active agent thread so we can clean up before starting a new session
    active_agent = {"thread": None, "room_id": None}

    class StartSessionRequest(BaseModel):
        roomId: str
        interviewId: str | None = None
        authToken: str

    @app.post("/start-session")
    async def start_session_endpoint(req: StartSessionRequest, background_tasks: BackgroundTasks):
        """Start an agent session in a VideoSDK room."""
        print(f"[HTTP API] Received start-session request: roomId={req.roomId}, interviewId={req.interviewId}")

        # Wait for previous agent thread to finish if still running
        if active_agent["thread"] and active_agent["thread"].is_alive():
            prev_room = active_agent["room_id"]
            print(f"[HTTP API] Previous session (room={prev_room}) still active, waiting for cleanup...", flush=True)
            active_agent["thread"].join(timeout=5)
            if active_agent["thread"].is_alive():
                print(f"[HTTP API] Previous session did not stop in time, proceeding anyway", flush=True)

        def run_agent():
            # FastAPI BackgroundTasks runs in a worker thread without an event loop.
            # WorkerJob / JobContext needs one, so create and set a new loop.
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                context_fn = lambda: make_context(
                    room_id=req.roomId,
                    auth_token=req.authToken,
                    interview_id=req.interviewId
                )
                job = WorkerJob(entrypoint=start_session, jobctx=context_fn)
                job.start()
            except Exception as e:
                print(f"[HTTP API] Error starting agent: {e}", flush=True)
            finally:
                loop.close()

        # Run in a dedicated thread instead of BackgroundTasks so we can track it
        agent_thread = threading.Thread(target=run_agent, daemon=True)
        active_agent["thread"] = agent_thread
        active_agent["room_id"] = req.roomId
        agent_thread.start()

        return {
            "status": "started",
            "roomId": req.roomId,
            "message": "Agent is joining the room"
        }

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    port = int(os.getenv("AGENT_HTTP_PORT", "8000"))
    print(f"[Agent] Starting HTTP server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    mode = os.getenv("AGENT_MODE", "standalone")

    if mode == "server":
        # Run as HTTP server that accepts session requests
        run_http_server()
    else:
        # Run as standalone agent (original behavior)
        job = WorkerJob(
            entrypoint=start_session,
            jobctx=lambda: make_context()
        )
        job.start()
