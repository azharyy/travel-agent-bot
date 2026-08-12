import json
import logging
from datetime import datetime, timezone
import ollama

from backend.core.config import settings
from backend.core.schemas import LivePriceData

logger = logging.getLogger(__name__)

class DataCleaner:
    """
    Agent 2: Local AI Cleaner. Parses compressed web markup into structured JSON 
    matching the exact LivePriceData contract, running entirely locally via Ollama.
    """
    def __init__(self):
        self.model = settings.ollama_cleaner_model
        # FIX: Changed base_url= to host= to match the Ollama SDK contract
        self.client = ollama.AsyncClient(host=settings.ollama_base_url)

    async def clean(self, property_id: str, compressed_html: str) -> LivePriceData:
        """
        Processes unformatted token-optimized text blocks and extracts critical metrics.
        Instructs the local LLM to output valid JSON text only.
        """
        logger.info(f"Agent 2 parsing extraction payloads for property: {property_id}")

        if "[ERROR]" in compressed_html:
            logger.warning(f"Skipping LLM parsing due to upstream scraper failure: {compressed_html}")
            return LivePriceData(
                property_id=property_id,
                scrape_timestamp=datetime.now(timezone.utc).isoformat(),
                is_available=False,
                availability_notes=f"Data collection error encountered: {compressed_html}"
            )

        system_prompt = (
            "You are an expert data parsing assistant. Your task is to extract property details "
            "from the clean text block provided and return a single, strictly compliant JSON object. "
            "Do not include any chat commentary, introduction, or markdown wraps (like ```json). "
            "\n\nExpected JSON Structure fields:\n"
            "{\n"
            '  "nightly_price_usd": float or null,\n'
            '  "total_price_usd": float or null,\n'
            '  "currency_original": string or null,\n'
            '  "cleaning_fee_usd": float or null,\n'
            '  "is_available": boolean or null,\n'
            '  "availability_notes": string or null\n'
            "}\n\n"
            "Rules:\n"
            "1. Extract the primary or currently active price per night listed.\n"
            "2. If prices are mentioned in other currencies (like EUR or EGP), convert roughly to USD or map original if distinct.\n"
            "3. If explicitly sold out, flag is_available as false."
        )

        user_content = f"Target Document Body:\n{compressed_html}"

        try:
            # Call local Ollama runtime asynchronously
            response = await self.client.generate(
                model=self.model,
                system=system_prompt,
                prompt=user_content,
                options={"temperature": 0.0}  # Hard deterministic parsing constraints
            )

            response_text = response.get("response", "").strip()
            
            # Clean up accidental markdown wraps if the local model appended them anyway
            if response_text.startswith("```"):
                response_text = response_text.strip("`").replace("json", "", 1).strip()

            # OPTIMIZATION: Parse and validate directly using Pydantic's native engine.
            # This handles missing fields, conversions, and validation logic instantly.
            live_data = LivePriceData.model_validate_json(response_text)
            
            # Inject dynamic runtime properties using model_copy
            return live_data.model_copy(update={
                "property_id": property_id,
                "scrape_timestamp": datetime.now(timezone.utc).isoformat()
            })

        except (json.JSONDecodeError, ValueError) as je:
            logger.error(f"Ollama output did not return structured valid JSON: {str(je)}. Text: {response_text}")
            return self._generate_fallback_data(property_id, f"JSON structural interpretation failure: {str(je)}")
        except Exception as e:
            logger.error(f"Agent 2 extraction runtime failure for {property_id}: {str(e)}")
            return self._generate_fallback_data(property_id, f"Extraction engine exception: {str(e)}")

    def _generate_fallback_data(self, property_id: str, reason: str) -> LivePriceData:
        return LivePriceData(
            property_id=property_id,
            scrape_timestamp=datetime.now(timezone.utc).isoformat(),
            is_available=None,
            availability_notes=f"Fallback triggered: {reason}"
        )