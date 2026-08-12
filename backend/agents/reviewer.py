# backend/agents/reviewer.py
import json
import logging
from datetime import datetime, timezone
from google import genai
from google.genai import types

from backend.core.config import settings
from backend.core.schemas import AgentReviewRequest, ValidationResult, LivePriceData

logger = logging.getLogger(__name__)

class ReviewerAgent:
    """
    Agent 3: Final Validator Gatekeeper. Evaluates combined static DB and 
    live dynamic payloads against explicit user preference filters.
    Utilizes Gemini via the modern google-genai SDK.
    """
    def __init__(self):
        # Initializes the standard modern GenAI client matching settings keys
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = "gemini-2.5-flash"

    async def review_property(self, request: AgentReviewRequest) -> ValidationResult:
        """
        Runs a deep compatibility audit across static and dynamic data sets
        to enforce hard business rules and verify constraints.
        """
        logger.info(f"Agent 3 auditing constraints evaluation for property: {request.property_id}")

        # Construct explicit system context guidance
        system_instruction = (
            "You are an elite Travel Validation Inspector. Your job is to perform a strict "
            "compliance audit comparing a real-estate property match against a traveler's explicit constraints.\n\n"
            "Analyze both the static file history and live updated details to determine structural fit. "
            "Output your analysis strictly inside the requested structural JSON schema context."
        )

        user_prompt = (
            f"--- Traveler Requirements ---\n{json.dumps(request.user_preferences, indent=2)}\n\n"
            f"--- Static Database Record ---\n{json.dumps(request.static_db_data, indent=2)}\n\n"
            f"--- Live Hydration State ---\n{json.dumps(request.live_scraped_data.model_dump(), indent=2)}\n\n"
            "Task: Verify if this property remains a sound match based on pricing, availability, and features. "
            "Flag explicit discrepancies (e.g., if live price exceeds budget caps, or if it is marked unavailable)."
        )

        try:
            # Request clean structured Pydantic mapping from Gemini natively
            response = self.client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=ValidationResult,
                    temperature=0.1,  # Keep reasoning steady and logical
                ),
            )

            # Native Pydantic validation directly from the raw JSON string text payload.
            result = ValidationResult.model_validate_json(response.text)
            
            # Dynamically merge or update the runtime property_id into the typed object
            return result.model_copy(update={"property_id": request.property_id})

        except Exception as e:
            logger.error(f"Agent 3 runtime evaluation exception for {request.property_id}: {str(e)}")
            return ValidationResult(
                property_id=request.property_id,
                is_valid_match=False,
                confidence_score=0.0,
                reasoning_summary=f"Validation failed during processing execution: {str(e)}",
                actionable_alerts=["VALIDATION_ERROR_TRIGGERED"]
            )

    async def review(
        self,
        properties: list,
        live_prices: dict[str, LivePriceData],
        user_message: str,
        max_budget_usd: float | None = None,
        distance_info: dict | None = None
    ) -> tuple[list, str]:
        """
        Batch processes a collection of property options, lining up with the 
        exact multi-argument signature invoked inside Stage 4 of the central Orchestrator.
        """
        logger.info(f"Agent 3 batch-reviewing processing pipeline for {len(properties)} properties.")
        verified_properties = []
        
        # Structure the target evaluation metrics context 
        user_preferences = {
            "user_message": user_message,
            "max_budget_usd": max_budget_usd
        }

        for prop in properties:
            # 1. Resolve or fall back the conditional dynamic Live Pricing data
            live_scraped = live_prices.get(prop.property_id)
            if not live_scraped:
                # Extract target value safely matching schema attributes
                fallback_price = getattr(prop, 'avg_price_usd', None)
                if fallback_price is None:
                    fallback_price = getattr(prop, 'nightly_price_usd', None)

                # FIX: Convert datetime to a clean string format (.isoformat()) to satisfy schema constraints
                live_scraped = LivePriceData(
                    property_id=str(prop.property_id),
                    nightly_price_usd=fallback_price,
                    is_available=getattr(prop, 'is_available', True),
                    scrape_timestamp=datetime.now(timezone.utc).isoformat()
                )
            # 2. Package data context matrices into a single AgentReviewRequest model
            static_db_payload = {
                "name": getattr(prop, 'name', 'Unknown Property'),
                "city": getattr(prop, 'city', 'Unknown City'),
                "country": getattr(prop, 'country', 'Egypt'),
                "amenities": getattr(prop, 'amenities', [])
            }

            review_request = AgentReviewRequest(
                property_id=prop.property_id,
                user_preferences=user_preferences,
                static_db_data=static_db_payload,
                live_scraped_data=live_scraped
            )

            # 3. Fire audit checks against the underlying evaluation system model
            validation = await self.review_property(review_request)

            # 4. Extract model data structural dictionary elements to safely mutate with metadata
            prop_dict = prop.model_dump() if hasattr(prop, "model_dump") else prop.__dict__.copy()
            
            # Map parameters perfectly according to VerifiedProperty Pydantic schema expectations
            prop_dict["passed_review"] = validation.is_valid_match
            prop_dict["reviewer_notes"] = validation.reasoning_summary
            prop_dict["distance_info"] = distance_info  # Injects geolocation telemetry metrics if generated

            # Defensively ensure keys are present for frontend UI mapping
            if "nightly_price_usd" not in prop_dict or prop_dict["nightly_price_usd"] is None:
                prop_dict["nightly_price_usd"] = prop_dict.get("avg_price_usd", 0.0)

            verified_properties.append(prop_dict)

        # 5. Synthesize clean conversational context messages responding back to the Orchestrator loop
        if distance_info:
            origin_name = distance_info.get("origin_name", "origin").title()
            dest_name = distance_info.get("dest_name", "destination").title()
            distance_km = distance_info.get("distance_km", "unknown")
            distance_text = f"The distance between {origin_name} and {dest_name} is approximately {distance_km} km."
        else:
            distance_text = ""

        # Compose professional conversational summary framing
        if any(p["passed_review"] for p in verified_properties):
            summary_message = (
                f"I discovered matching listings for your travel request. {distance_text} "
                f"I ran an automated quality audit checking them against your preferences. Here are the top properties "
                f"retaining live availability parameters:"
            )
        else:
            summary_message = (
                f"I processed your query and found properties, but they may have strict budget cap mismatches. "
                f"{distance_text} Review the details below for validation reasoning alerts:"
            )

        return verified_properties, summary_message