# GuideMe

GuideMe is an Egypt hotel recommendation app. The current active flow is:

1. Use a local static hotel catalog for the first recommendations.
2. Store and search that catalog in ChromaDB using CLIP embeddings.
3. Let the user select a hotel.
4. Only after selection, collect exact dates and guest details.
5. Call the external RollingGo MCP server for live room availability.

GuideMe acts as an MCP client. It does not need to run its own MCP server for
the normal app flow.

## Tech Stack

- Backend: FastAPI
- Frontend: Streamlit
- Agent orchestration: LangGraph
- LLM: Gemini through `langchain-google-genai`
- Static retrieval database: ChromaDB
- Static hotel embeddings: `sentence-transformers/clip-ViT-B-32`
- External live hotel provider: RollingGo MCP
- Dependency management: `pyproject.toml` with `uv`

## Quick Start

Install dependencies:

```powershell
uv sync
```

Create a `.env` file from `.env.example`, then set at least:

```env
GEMINI_API_KEY=your_gemini_key
ROLLINGGO_MCP_URL=https://mcp.rollinggo.cn/mcp
ROLLINGGO_API_KEY=your_rollinggo_key
```

Build the static hotel JSON database:

```powershell
python -m backend.hotel_collector.build_hotels_db
```

Ingest static hotels into Chroma:

```powershell
python -m backend.hotel_collector.ingest_chroma --reset-collection --skip-images
```

Run the backend:

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Run the frontend:

```powershell
python -m streamlit run frontend/app.py --server.address 127.0.0.1 --server.port 8501
```

Open:

```text
http://127.0.0.1:8501
```

## Backend Overview

The backend has three main jobs:

1. Serve the chat API used by the Streamlit app.
2. Run the hotel conversation state machine.
3. Search local hotel recommendations first, then call RollingGo only for live
   availability after a hotel is selected.

The important design rule is that initial recommendations are not live pricing
results. They come from the local `egypt_hotels` Chroma collection. Live price
and room data only appears after the user chooses one hotel and gives dates,
adults, children, child ages if needed, and room count.

## Request Flow

### 1. User enters a request in Streamlit

The user types something like:

```text
Beach resorts in Hurghada
```

or clicks a hotel card button such as:

```text
option 2
```

The frontend sends a POST request to:

```text
POST /api/chat
```

with this shape:

```json
{
  "session_id": "streamlit-session-id",
  "user_message": "Beach resorts in Hurghada",
  "conversation_history": []
}
```

### 2. FastAPI receives the request

`backend/main.py` creates the FastAPI app and mounts the chat router under
`/api`.

`backend/api/chat_router.py` receives `/api/chat`, validates the request with
`ChatRequest`, then forwards it to the application orchestrator stored in
`app.state.orchestrator`.

### 3. The orchestrator chooses the active agent

`backend/agents/orchestrator.py` keeps one `HotelGraphAgent` per `session_id`.
This gives each frontend session a lightweight in-memory conversation state.

There is no database-backed session store right now. If the backend restarts,
chat state is lost and the user starts a new conversation.

### 4. LangGraph hotel agent runs the conversation

`backend/agents/hotel_graph_agent.py` owns the active hotel state machine.

It tracks:

- city
- hotel preference/type
- query rewrites
- local RAG results
- shown offset for "show more"
- selected hotel
- check-in/check-out dates
- adults
- children count and ages
- room count
- live availability
- next pending question

The agent first uses deterministic parsing for common inputs:

- cities such as Cairo, Hurghada, Sharm El-Sheikh, North Coast, and Sahel
- hotel types such as beach, luxury, family, budget, romantic, business, nature,
  and spa
- option selection by number, ordinal, or hotel name
- exact dates
- adults, children, child ages, and rooms
- "show more"

Gemini structured output is available as a fallback for intent extraction, but
the main flow does not depend on LLM query rewriting.

### 5. Local RAG search returns initial hotel cards

When the agent has a city and hotel preference, it calls:

```python
search_hotels(location, hotel_type)
```

from `backend/agents/hotel_tools.py`.

That function:

- canonicalizes the city
- creates one original query plus three deterministic rewrites
- embeds the queries with `sentence-transformers/clip-ViT-B-32`
- searches the `egypt_hotels` Chroma collection with `query_embeddings`
- deduplicates by provider hotel id
- boosts exact city matches
- returns the first display page and keeps up to 20 options for "show more"

The returned initial cards include:

- option number
- name
- city
- hotel type
- summary
- tags
- amenities
- image URL
- internal hotel id

The initial recommendation text and cards do not show booking links.

### 6. User selects a hotel

The user can select by typing:

```text
2
option 2
second
```

or by clicking the button under a hotel card in the frontend.

The agent stores the selected hotel and starts collecting live lookup details.

### 7. Agent collects live lookup details

Before calling RollingGo live detail, the agent asks for any missing required
fields:

- check-in date
- check-out date
- total adults
- children count
- child ages, only if children count is greater than zero
- room count

Dates are normalized to ISO format. For example, `2026-6-20` becomes
`2026-06-20`.

### 8. RollingGo live availability is called

Once all live lookup fields are present, `hotel_graph_agent.py` calls:

```python
fetch_live_pricing(...)
```

from `backend/agents/hotel_tools.py`.

`fetch_live_pricing` validates the input and then calls:

```python
RollingGoMCPClient().get_hotel_detail(payload)
```

The payload sent to RollingGo includes:

- hotel id
- check-in/check-out
- adults per room
- child count and ages
- room count
- country code
- currency

The response is normalized into live room options:

- room name
- rate plan name
- price
- currency
- inventory count
- cancellation summary
- booking URL

Booking URLs are only shown after this live lookup step.

## Frontend Overview

The frontend is a Streamlit app in `frontend/app.py`.

It handles:

- page layout and styling
- session id creation
- chat message storage in `st.session_state`
- calls to `POST /api/chat`
- rendering static hotel cards
- rendering live availability cards
- click-to-select buttons for hotel options
- dynamic input placeholders based on the current follow-up question

`frontend/components/property_card.py` renders:

- static recommendation cards
- "Check availability for option X" buttons
- live room availability cards
- post-live booking links

Initial hotel cards intentionally hide booking links. They are only displayed
after RollingGo returns live availability.

## Main Backend Files

### `backend/main.py`

FastAPI application entrypoint.

Responsibilities:

- creates the FastAPI app
- configures CORS for Streamlit
- initializes Chroma sample data for the legacy collection if needed
- creates the `Orchestrator`
- stores it on `app.state`
- includes the chat router under `/api`
- exposes `/health`

The app is started with:

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

### `backend/api/chat_router.py`

Defines the `/api/chat` endpoint.

Responsibilities:

- accepts `ChatRequest`
- retrieves `request.app.state.orchestrator`
- calls `orchestrator.run(...)`
- returns `ChatResponse`
- turns startup/orchestrator errors into HTTP errors

### `backend/agents/orchestrator.py`

Session-level agent manager.

Responsibilities:

- creates one `HotelGraphAgent` per `session_id`
- keeps state in memory
- calls the hotel agent
- adapts the result into `ChatResponse`

### `backend/agents/hotel_graph_agent.py`

The main hotel conversation state machine.

Responsibilities:

- parse city, hotel preference, selections, dates, guests, and rooms
- ask the next missing question
- call local hotel RAG search before live lookup
- support "show more"
- call live pricing only after a hotel is selected and trip details are complete
- return `assistant_message`, `properties`, `live_availability`, and
  `pipeline_stage_reached`

Important pipeline stages include:

- `need_city`
- `need_preference`
- `rag_recommendations`
- `show_more`
- `awaiting_hotel_selection`
- `collecting_live_details`
- `live_availability`
- `rag_search_empty`
- `setup_error`

### `backend/agents/hotel_tools.py`

Tool functions used by the hotel agent.

Responsibilities:

- search local Chroma recommendations with `search_hotels`
- keep a compatibility wrapper named `search_properties`
- embed user queries with CLIP
- merge and rank Chroma results
- validate live lookup inputs
- call RollingGo MCP for live detail
- normalize live room/rate plans

This file is the boundary between the agent and retrieval/provider tools.

### `backend/agents/query_rewriter.py`

Deterministic query rewrite helper.

Responsibilities:

- returns exactly three query rewrites for a city and hotel type
- uses fixed synonym groups such as luxury, beach, family, budget, romantic,
  business, nature, and spa

This keeps local retrieval predictable and avoids LLM query rewriting.

### `backend/core/config.py`

Application settings.

Responsibilities:

- loads `.env` using Pydantic Settings
- defines API host/port
- defines Gemini settings
- defines RollingGo MCP settings
- defines hotel catalog build settings
- defines Chroma collection paths and names
- defines legacy settings that are still present but not part of the active
  hotel MCP flow

Important hotel settings:

```env
GEMINI_MODEL=gemini-2.5-flash
ROLLINGGO_MCP_URL=https://mcp.rollinggo.cn/mcp
ROLLINGGO_API_KEY=
ROLLINGGO_ACCEPT_LANGUAGE=en_US
HOTEL_COUNTRY_CODE=EG
HOTEL_CURRENCY=EGP
CHROMA_DB_PATH=backend/chroma_db
HOTEL_COLLECTION_NAME=egypt_hotels
```

### `backend/core/database.py`

Chroma client helpers.

Responsibilities:

- `get_chroma_client()` returns the hotel Chroma client at `CHROMA_DB_PATH`
- `get_hotel_collection_name()` returns `HOTEL_COLLECTION_NAME`
- legacy helpers still manage the older `travel_properties` collection
- sample legacy data is seeded for old flows and health checks

The active hotel flow uses the `egypt_hotels` collection and does not mix with
the old `travel_properties` embeddings.

### `backend/core/schemas.py`

Pydantic API and legacy data schemas.

Responsibilities:

- validates `ChatRequest`
- defines `ChatResponse`
- keeps older property, geocoding, hydration, price, and validation schemas

The active frontend expects `ChatResponse` to include:

- `assistant_message`
- `properties`
- `live_availability`
- `pipeline_stage_reached`

## Hotel Collector Files

The hotel collector builds the local static catalog used by RAG.

### `backend/hotel_collector/cities.py`

Defines the supported Egypt destination queries.

Examples:

- Cairo hotels
- Hurghada hotels
- Sharm El Sheikh hotels
- North Coast Egypt hotels
- Dahab hotels
- Ain Sokhna hotels

### `backend/hotel_collector/models.py`

Pydantic models for hotel data.

Important model:

```python
HotelCard
```

`HotelCard` stores only static recommendation data. It intentionally does not
store:

- latitude
- longitude
- live price
- live availability
- huge raw provider payloads

### `backend/hotel_collector/utils.py`

Normalization helpers.

Responsibilities:

- slug generation
- city canonicalization
- Sahel to North Coast normalization
- Sharm El Sheikh to Sharm El-Sheikh normalization
- tag and amenity normalization
- safe numeric parsing
- stable local hotel id creation
- Chroma metadata flattening
- RAG summary creation

### `backend/hotel_collector/rollinggo_mcp_client.py`

Async client for the external RollingGo MCP server.

Responsibilities:

- loads RollingGo URL, API key, and language from settings or `.env`
- configures `langchain_mcp_adapters.client.MultiServerMCPClient`
- lists available tools
- calls:
  - `searchHotels`
  - `getHotelDetail`
  - `getHotelSearchTags`
- handles missing config, connection failures, tool-not-found errors, provider
  errors, and different response shapes
- masks API keys in error messages

### `backend/hotel_collector/test_mcp_connection.py`

Manual connectivity test for RollingGo MCP.

Run:

```powershell
python -m backend.hotel_collector.test_mcp_connection
```

It lists tools, tries tags, and performs a small Hurghada search without saving
data.

### `backend/hotel_collector/build_hotels_db.py`

Manual static hotel database builder.

Run all cities:

```powershell
python -m backend.hotel_collector.build_hotels_db
```

Run one city:

```powershell
python -m backend.hotel_collector.build_hotels_db --city "Hurghada"
```

Responsibilities:

- calls RollingGo `searchHotels` for static discovery only
- optionally caches `getHotelSearchTags` to
  `backend/data/rollinggo_tags.json`
- loops over supported cities
- runs multiple search profiles per city
- normalizes provider data into `HotelCard`
- deduplicates by provider hotel id
- ranks richer and more relevant cards
- saves pretty UTF-8 JSON to `backend/data/hotels.json`

It does not store live availability, static prices as truth, latitude/longitude,
or raw provider payloads.

### `backend/hotel_collector/ingest_chroma.py`

Ingests `backend/data/hotels.json` into Chroma.

Run:

```powershell
python -m backend.hotel_collector.ingest_chroma --reset-collection --skip-images
```

Responsibilities:

- reads static `HotelCard` entries
- builds document text with `summarize_hotel_for_rag`
- embeds text with `sentence-transformers/clip-ViT-B-32`
- optionally downloads and embeds hotel images
- fuses text and image vectors when enabled
- writes to the `egypt_hotels` collection with upsert

Repeated ingestion updates existing ids instead of creating duplicates.

### `backend/hotel_collector/test_retrieval.py`

Manual local retrieval test.

Run:

```powershell
python -m backend.hotel_collector.test_retrieval
```

It embeds a sample query and prints the top hotel matches from Chroma.

## Data Directories

### `backend/data/hotels.json`

Static hotel cards built from RollingGo `searchHotels`.

This file is the source data for Chroma ingestion.

### `backend/data/rollinggo_tags.json`

Optional cache of RollingGo search tags.

### `backend/data/hotel_images/`

Optional cache for downloaded hotel images used during image embedding.

### `backend/chroma_db/`

Persistent ChromaDB directory for the active `egypt_hotels` collection.

### `data/chromadb/`

Legacy ChromaDB directory for the older `travel_properties` collection.

## Legacy Files Still Present

Some files remain from the original travel-agent prototype:

- `backend/agents/gemini_searcher.py`
- `backend/agents/data_cleaner.py`
- `backend/agents/reviewer.py`
- `backend/scrapers/scrapling_engine.py`
- `backend/scrapers/html_compressor.py`
- legacy functions in `backend/core/database.py`

These are not the active hotel recommendation path. The current hotel flow does
not use direct Booking.com scraping and does not use Ollama for the hotel RAG
or live availability path.

`backend/rollinggo_mcp_server.py` may exist in the repo from experiments, but
the normal GuideMe app is an MCP client and should use the external RollingGo
MCP URL configured in `.env`.

## Configuration Flow

Settings are loaded from `.env` through `backend/core/config.py`.

The frontend also imports `settings`, mainly to know the backend URL:

```env
API_BASE_URL=http://127.0.0.1:8000
```

The backend reads:

- Gemini settings for the LangGraph agent
- RollingGo settings for MCP calls
- hotel catalog defaults for builder scripts
- Chroma paths and collection names

## Current Active End-to-End Flow

Here is the full code path for a normal user interaction:

```text
frontend/app.py
  -> POST /api/chat
backend/api/chat_router.py
  -> Orchestrator.run(...)
backend/agents/orchestrator.py
  -> HotelGraphAgent.arun(...)
backend/agents/hotel_graph_agent.py
  -> parse intent
  -> ask for city/preference if missing
  -> search_hotels(...) when city and preference are known
backend/agents/hotel_tools.py
  -> embed query with CLIP
  -> query Chroma egypt_hotels
  -> return static recommendations
frontend/components/property_card.py
  -> render cards and option buttons
frontend/app.py
  -> user selects option
backend/agents/hotel_graph_agent.py
  -> collect dates, guests, children, rooms
backend/agents/hotel_tools.py
  -> fetch_live_pricing(...)
backend/hotel_collector/rollinggo_mcp_client.py
  -> call RollingGo getHotelDetail
backend/agents/hotel_tools.py
  -> normalize live rooms
frontend/components/property_card.py
  -> show live rooms and booking link
```

## Response Shape

The frontend expects every chat response to include:

```json
{
  "assistant_message": "Text to show in the chat",
  "properties": [],
  "live_availability": null,
  "pipeline_stage_reached": "rag_recommendations"
}
```

`properties` is used for static hotel cards.

`live_availability` is used only after RollingGo live detail lookup.

## Common Commands

Check RollingGo MCP connectivity:

```powershell
python -m backend.hotel_collector.test_mcp_connection
```

Build hotel JSON:

```powershell
python -m backend.hotel_collector.build_hotels_db
```

Build one city:

```powershell
python -m backend.hotel_collector.build_hotels_db --city "Hurghada"
```

Ingest into Chroma:

```powershell
python -m backend.hotel_collector.ingest_chroma --reset-collection --skip-images
```

Test retrieval:

```powershell
python -m backend.hotel_collector.test_retrieval
```

Run backend:

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Run frontend:

```powershell
python -m streamlit run frontend/app.py --server.address 127.0.0.1 --server.port 8501
```

## Notes For Future Development

- Keep initial recommendations local and static.
- Do not show booking URLs before live lookup.
- Do not mix `egypt_hotels` CLIP embeddings with legacy `travel_properties`
  embeddings.
- Do not treat static provider prices as live truth.
- Do not store raw provider payloads in `hotels.json`.
- Keep RollingGo API keys out of logs.
- If the Chroma collection reports embedding dimension errors, reset and ingest
  the `egypt_hotels` collection again.
