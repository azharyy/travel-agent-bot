from bs4 import BeautifulSoup
import re

def compress_html(raw_html: str, max_chars: int = 8000) -> str:
    """
    Strips raw HTML down to text content optimized for price and property extraction.
    Safeguards short numerical/currency chunks while dropping layout noise.
    """
    if not raw_html:
        return ""

    # Using 'html.parser' as standard default; switch to 'lxml' if installed in your env
    soup = BeautifulSoup(raw_html, "html.parser")

    # Remove code blocks and non-content navigation shells entirely
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "svg"]):
        tag.decompose()

    # Extract all text fragments separated by a clean newline to isolate blocks
    text = soup.get_text(separator="\n")

    cleaned_lines = []
    # Process line-by-line instead of splitting by periods
    for line in text.splitlines():
        line = line.strip()
        
        # Skip empty lines completely
        if not line:
            continue
            
        # KEEP the line if it meets any of these criteria:
        # 1. It contains descriptive content (> 20 characters)
        # 2. It contains numbers or currency indicators (e.g., "$150", "4.9", "2026")
        if len(line) > 20 or re.search(r"[\d$€£¥]", line):
            # Collapse internal multi-spaces or tabs within the line
            line = re.sub(r"\s+", " ", line)
            cleaned_lines.append(line)

    # Join with newlines so structural separations remain clear to the LLM
    compressed_text = "\n".join(cleaned_lines)

    # Truncate to safeguard Phi-3 Mini's context constraint
    if len(compressed_text) > max_chars:
        compressed_text = compressed_text[:max_chars] + "...[truncated]"

    return compressed_text.strip()