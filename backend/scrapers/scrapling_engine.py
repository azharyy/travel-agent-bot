# backend/scrapers/scrapling_engine.py
from datetime import datetime, timezone
import logging
from scrapling import Fetcher  # Safe stealth fetcher engine

from backend.core.schemas import HydrationRequest, RawHydrationData
from backend.scrapers.html_compressor import compress_html

logger = logging.getLogger(__name__)

class ScraplingEngine:
    """
    Executes high-stealth web scraping requests for live pricing and 
    availability updates, instantly compressing data to minimize LLM overhead.
    """
    def __init__(self):
        # We can configure custom headers or regional proxies here if necessary
        pass

    async def hydrate(self, request: HydrationRequest) -> RawHydrationData:
        """
        Fetches live web content for a target booking URL and returns 
        token-optimized HTML payload ready for processing.
        """
        logger.info(f"Initiating live hydration capture for property: {request.property_id}")
        
        try:
            # Fetch content natively using Scrapling's specialized fetchers
            # It uses anti-bot fingerprint bypasses automatically
            fetcher = Fetcher(url=request.source_url)
            
            raw_html = fetcher.content
            if not raw_html:
                raise ValueError("Received empty HTML content response from target url.")

            # Pass the raw markup straight through your newly optimized token compressor
            # Using standard 8000 character limits for local model bounds
            optimized_text = compress_html(raw_html, max_chars=8000)

            return RawHydrationData(
                property_id=request.property_id,
                source_url=request.source_url,
                compressed_html=optimized_text,
                scrape_timestamp=datetime.now(timezone.utc).isoformat()
            )

        except Exception as e:
            logger.error(f"Hydration pipeline failure for property {request.property_id}: {str(e)}")
            # Fallback gracefully with an error indicator inside the context string 
            # to let downstream agents explicitly notice missing data fields.
            return RawHydrationData(
                property_id=request.property_id,
                source_url=request.source_url,
                compressed_html=f"[ERROR] Live scraping failed: {str(e)}",
                scrape_timestamp=datetime.now(timezone.utc).isoformat()
            )