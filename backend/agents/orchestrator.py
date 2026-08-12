"""Application orchestrator for GuideMe hotel conversations."""

from __future__ import annotations

import logging

from backend.agents.hotel_graph_agent import HotelGraphAgent
from backend.core.config import Settings
from backend.core.schemas import ChatRequest, ChatResponse


logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Lightweight session orchestrator.

    GuideMe is now an MCP client for hotel live availability. Initial hotel
    recommendations come from local Chroma through HotelGraphAgent, while
    RollingGo detail calls happen only after a hotel/date/guest selection.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._hotel_agents: dict[str, HotelGraphAgent] = {}

    def _agent_for_session(self, session_id: str) -> HotelGraphAgent:
        if session_id not in self._hotel_agents:
            self._hotel_agents[session_id] = HotelGraphAgent()
        return self._hotel_agents[session_id]

    async def run(self, request: ChatRequest) -> ChatResponse:
        logger.info("Processing hotel chat session %s", request.session_id)
        try:
            agent = self._agent_for_session(request.session_id)
            result = await agent.arun(request.user_message)
            return ChatResponse(
                session_id=request.session_id,
                assistant_message=result.get("assistant_message", ""),
                properties=result.get("properties", []),
                live_availability=result.get("live_availability"),
                pipeline_stage_reached=result.get("pipeline_stage_reached", "unknown"),
            )
        except Exception as exc:
            logger.exception("Hotel graph failed for session %s", request.session_id)
            return ChatResponse(
                session_id=request.session_id,
                assistant_message="Something went wrong processing your hotel request. Please try again.",
                properties=[],
                live_availability=None,
                pipeline_stage_reached="error",
                error=str(exc),
            )
