"""Async MCP client for the external RollingGo hotel server."""

from __future__ import annotations

import inspect
import json
import os
import re
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency is declared in pyproject.
    load_dotenv = None


SEARCH_HOTELS_TOOL = "searchHotels"
GET_HOTEL_DETAIL_TOOL = "getHotelDetail"
GET_HOTEL_SEARCH_TAGS_TOOL = "getHotelSearchTags"


class RollingGoMCPError(RuntimeError):
    """Base error for RollingGo MCP client failures."""


class RollingGoMCPConfigurationError(RollingGoMCPError):
    """Raised when client configuration is missing or invalid."""


class RollingGoMCPConnectionError(RollingGoMCPError):
    """Raised when the external MCP server cannot be reached."""


class RollingGoMCPToolNotFoundError(RollingGoMCPError):
    """Raised when the expected MCP tool is not advertised by RollingGo."""


class RollingGoMCPToolError(RollingGoMCPError):
    """Raised when a RollingGo MCP tool call fails."""


def _load_dotenv_once() -> None:
    if load_dotenv is not None:
        load_dotenv()


def _load_project_settings() -> Any | None:
    try:
        from backend.core.config import get_settings

        return get_settings()
    except Exception:
        return None


def _config_value(settings: Any | None, attr: str, env_name: str, default: str = "") -> str:
    if settings is not None:
        value = getattr(settings, attr, None)
        if value is not None:
            return str(value).strip()
    return os.getenv(env_name, default).strip()


def _mask_secret(secret: str) -> str:
    if not secret:
        return "<missing>"
    if len(secret) <= 8:
        return "<set>"
    return f"{secret[:4]}...{secret[-4:]}"


class RollingGoMCPClient:
    """Thin async client for RollingGo's externally managed MCP server."""

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        accept_language: str | None = None,
        transport: str = "streamable_http",
    ) -> None:
        _load_dotenv_once()
        project_settings = _load_project_settings()

        self.url = (
            url
            or _config_value(
                project_settings,
                "rollinggo_mcp_url",
                "ROLLINGGO_MCP_URL",
                "http://127.0.0.1:8010/mcp",
            )
        ).strip()
        self.api_key = (
            api_key
            or _config_value(project_settings, "rollinggo_api_key", "ROLLINGGO_API_KEY", "")
        ).strip()
        self.accept_language = (
            accept_language
            or _config_value(
                project_settings,
                "rollinggo_accept_language",
                "ROLLINGGO_ACCEPT_LANGUAGE",
                "en_US",
            )
        ).strip()
        # langchain-mcp-adapters exposes HTTP MCP as "streamable_http" in current versions.
        self.transport = transport
        self._tools: list[Any] | None = None

        if not self.url:
            raise RollingGoMCPConfigurationError("ROLLINGGO_MCP_URL is required.")
        if not self.api_key:
            raise RollingGoMCPConfigurationError("ROLLINGGO_API_KEY is required.")

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Accept-Language": self.accept_language or "en_US",
        }

    def _server_config(self, transport: str) -> dict[str, dict[str, Any]]:
        return {
            "rollinggo": {
                "url": self.url,
                "transport": transport,
                "headers": self.headers,
            }
        }

    async def list_tools(self) -> list[dict[str, Any]]:
        """List tools advertised by the RollingGo MCP server."""
        tools = await self._get_tools()
        return [self._describe_tool(tool) for tool in tools]

    async def search_hotels(self, payload: dict[str, Any]) -> Any:
        """Call RollingGo searchHotels."""
        return await self._call_tool(SEARCH_HOTELS_TOOL, payload)

    async def get_hotel_detail(self, payload: dict[str, Any]) -> Any:
        """Call RollingGo getHotelDetail."""
        return await self._call_tool(GET_HOTEL_DETAIL_TOOL, payload)

    async def get_hotel_search_tags(self) -> Any:
        """Call RollingGo getHotelSearchTags."""
        return await self._call_tool(GET_HOTEL_SEARCH_TAGS_TOOL, {})

    async def _get_tools(self) -> list[Any]:
        if self._tools is None:
            self._tools = await self._load_tools_with_fallback()
        return self._tools

    async def _load_tools_with_fallback(self) -> list[Any]:
        try:
            return await self._load_tools(self.transport)
        except RollingGoMCPConfigurationError:
            raise
        except RollingGoMCPError:
            raise
        except Exception as exc:
            if self.transport == "http" and self._looks_like_transport_error(exc):
                # Current langchain-mcp-adapters versions use "streamable_http" for HTTP MCP.
                self.transport = "streamable_http"
                return await self._load_tools(self.transport)
            if self.transport == "streamable_http" and self._looks_like_transport_error(exc):
                self.transport = "http"
                return await self._load_tools(self.transport)
            raise self._connection_error(exc) from exc

    async def _load_tools(self, transport: str) -> list[Any]:
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
        except ImportError as exc:
            raise RollingGoMCPConfigurationError(
                "langchain-mcp-adapters is not installed. Install project dependencies first."
            ) from exc

        try:
            client = MultiServerMCPClient(self._server_config(transport))
            tools = await client.get_tools()
        except RollingGoMCPError:
            raise
        except Exception as exc:
            raise self._connection_error(exc) from exc

        if tools is None:
            return []
        return list(tools)

    async def _call_tool(self, tool_name: str, payload: dict[str, Any]) -> Any:
        tool = await self._find_tool(tool_name)
        try:
            result = await self._invoke_tool(tool, payload or {})
        except RollingGoMCPError:
            raise
        except Exception as exc:
            raise RollingGoMCPToolError(
                f"RollingGo MCP tool '{tool_name}' failed: {self._safe_error_message(exc)}"
            ) from exc
        normalized = self._normalize_provider_result(result)
        provider_error = self._provider_error_message(normalized)
        if provider_error:
            raise RollingGoMCPToolError(
                f"RollingGo MCP tool '{tool_name}' failed: {provider_error}"
            )
        return normalized

    async def _find_tool(self, tool_name: str) -> Any:
        tools = await self._get_tools()
        for tool in tools:
            candidate = self._tool_name(tool)
            if self._matches_tool_name(candidate, tool_name):
                return tool

        available = ", ".join(sorted(filter(None, (self._tool_name(tool) for tool in tools))))
        if not available:
            available = "<none>"
        raise RollingGoMCPToolNotFoundError(
            f"RollingGo MCP tool '{tool_name}' was not found. Available tools: {available}"
        )

    async def _invoke_tool(self, tool: Any, payload: dict[str, Any]) -> Any:
        if hasattr(tool, "ainvoke"):
            return await tool.ainvoke(payload)
        if hasattr(tool, "arun"):
            return await tool.arun(payload)
        if callable(tool):
            value = tool(payload)
            if inspect.isawaitable(value):
                return await value
            return value
        raise RollingGoMCPToolError(f"Tool '{self._tool_name(tool)}' is not callable.")

    def _connection_error(self, exc: Exception) -> RollingGoMCPConnectionError:
        return RollingGoMCPConnectionError(
            "Could not connect to the external RollingGo MCP server at "
            f"{self.url}. Confirm RollingGo is running separately and credentials are set "
            f"(API key {_mask_secret(self.api_key)}). Details: {self._safe_error_message(exc)}"
        )

    @staticmethod
    def _looks_like_transport_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return "transport" in text and (
            "http" in text or "streamable" in text or "unsupported" in text or "invalid" in text
        )

    def _safe_error_message(self, exc: Exception) -> str:
        nested = getattr(exc, "exceptions", None)
        if nested:
            messages = [self._safe_error_message(item) for item in nested]
            unique_messages = []
            for message in messages:
                if message and message not in unique_messages:
                    unique_messages.append(message)
            return "; ".join(unique_messages) or exc.__class__.__name__

        message = str(exc).strip()
        if self.api_key:
            message = message.replace(self.api_key, _mask_secret(self.api_key))
        message = re.sub(r"Bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer <redacted>", message)
        return message or exc.__class__.__name__

    @staticmethod
    def _tool_name(tool: Any) -> str:
        return str(getattr(tool, "name", None) or getattr(tool, "id", None) or "")

    @staticmethod
    def _matches_tool_name(candidate: str, expected: str) -> bool:
        if not candidate:
            return False
        candidate_lower = candidate.lower()
        expected_lower = expected.lower()
        return (
            candidate_lower == expected_lower
            or candidate_lower.endswith(f".{expected_lower}")
            or candidate_lower.endswith(f"_{expected_lower}")
        )

    @staticmethod
    def _describe_tool(tool: Any) -> dict[str, Any]:
        args_schema = getattr(tool, "args_schema", None)
        schema: Any = None
        if args_schema is not None:
            if hasattr(args_schema, "model_json_schema"):
                schema = args_schema.model_json_schema()
            elif isinstance(args_schema, Mapping):
                schema = dict(args_schema)

        return {
            "name": RollingGoMCPClient._tool_name(tool),
            "description": getattr(tool, "description", "") or "",
            "args_schema": schema,
        }

    @classmethod
    def _normalize_provider_result(cls, value: Any) -> Any:
        if value is None:
            return {}

        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return {}
            try:
                return cls._normalize_provider_result(json.loads(stripped))
            except json.JSONDecodeError:
                return stripped

        if isinstance(value, bytes):
            return cls._normalize_provider_result(value.decode("utf-8", errors="replace"))

        if isinstance(value, Mapping):
            for key in ("structuredContent", "structured_content", "json"):
                if key in value and value[key] is not None:
                    return cls._normalize_provider_result(value[key])

            if "content" in value and len(value) == 1:
                return cls._normalize_provider_result(value["content"])

            if "text" in value and set(value.keys()).issubset(
                {"type", "text", "mimeType", "annotations", "id"}
            ):
                return cls._normalize_provider_result(value["text"])

            return {str(key): cls._normalize_provider_result(item) for key, item in value.items()}

        if isinstance(value, list | tuple):
            normalized = [cls._normalize_provider_result(item) for item in value]
            if len(normalized) == 1 and cls._looks_like_content_item(value[0]):
                return normalized[0]
            return normalized

        if hasattr(value, "model_dump"):
            return cls._normalize_provider_result(value.model_dump())

        if is_dataclass(value):
            return cls._normalize_provider_result(asdict(value))

        if hasattr(value, "content"):
            return cls._normalize_provider_result(getattr(value, "content"))

        if hasattr(value, "text"):
            return cls._normalize_provider_result(getattr(value, "text"))

        if hasattr(value, "dict"):
            return cls._normalize_provider_result(value.dict())

        return value

    @staticmethod
    def _looks_like_content_item(value: Any) -> bool:
        if hasattr(value, "text") or hasattr(value, "content"):
            return True
        if not isinstance(value, Mapping):
            return False
        return bool({"text", "content"} & set(value.keys())) and set(value.keys()).issubset(
            {"type", "text", "content", "mimeType", "annotations", "id"}
        )

    @classmethod
    def _provider_error_message(cls, value: Any) -> str | None:
        if isinstance(value, str):
            text = value.strip()
            if text.lower().startswith("error executing tool"):
                return text
            return None

        if isinstance(value, Mapping):
            success = value.get("success")
            is_error = value.get("isError") or value.get("is_error")
            if success is False or is_error is True:
                error = value.get("error") or value.get("message") or value.get("text")
                return str(error or "Provider returned an error.").strip()

            if value.get("type") == "text" and isinstance(value.get("text"), str):
                text = value["text"].strip()
                if text.lower().startswith("error executing tool"):
                    return text
            return None

        if isinstance(value, list):
            messages = [cls._provider_error_message(item) for item in value]
            messages = [message for message in messages if message]
            if messages and len(messages) == len(value):
                unique_messages = []
                for message in messages:
                    if message not in unique_messages:
                        unique_messages.append(message)
                return "; ".join(unique_messages)

        return None
