import html
from urllib.parse import quote, urlparse, urlunparse

import streamlit as st


def _escape(value) -> str:
    return html.escape(str(value or ""))


def _tags_text(value) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value if item)
    return str(value or "")


def _clean_url(value) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if not text or text in {"{}", "[]", "null", "None", "none"}:
        return ""

    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""

    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            quote(parsed.path, safe="/:%"),
            parsed.params,
            quote(parsed.query, safe="=&?/:,+%"),
            parsed.fragment,
        )
    )


def render_property_card(prop: dict) -> str | None:
    """
    Render an initial static hotel recommendation.

    Booking URLs are intentionally not shown here. They are exposed only after
    live availability lookup returns room options.
    """
    is_hotel_card = "hotel_type" in prop or "summary" in prop
    if is_hotel_card:
        return _render_hotel_card(prop)

    _render_legacy_property_card(prop)
    return None


def _render_hotel_card(prop: dict) -> str | None:
    tags = prop.get("tags_text") or _tags_text(prop.get("tags"))
    amenities = prop.get("amenities_text") or _tags_text(prop.get("amenities"))
    option_number = prop.get("option_number")

    st.markdown(
        f"""
<div style="
    border: 1px solid #d8e0ea;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 16px;
    background: #ffffff;
    box-shadow: 0 2px 8px rgba(24, 37, 56, 0.06);
">
    <div style="display: flex; justify-content: space-between; gap: 16px; align-items: start;">
        <div>
            <h3 style="color: #172033; margin: 0;">{_escape(prop.get("option_number"))}. {_escape(prop.get("name"))}</h3>
            <p style="color: #516173; margin: 4px 0 8px 0;">
                {_escape(prop.get("city"))} &nbsp;|&nbsp; {_escape(prop.get("hotel_type"))}
            </p>
        </div>
    </div>
    <p style="color: #263445; margin: 6px 0;">{_escape(prop.get("summary"))}</p>
    <p style="color: #245f68; margin: 6px 0;"><strong>Tags:</strong> {_escape(tags or "Not listed")}</p>
    <p style="color: #516173; margin: 6px 0;"><strong>Amenities:</strong> {_escape(amenities or "Not listed")}</p>
</div>
""",
        unsafe_allow_html=True,
    )

    # Provider image URLs can expire or reject hotlinking; keep cards clean and
    # avoid standalone broken image icons during the recommendation phase.
    if option_number:
        key = f"hotel_select_{option_number}_{prop.get('hotel_id') or prop.get('name')}"
        if st.button(
            f"Check availability for option {option_number}",
            key=key,
            width="stretch",
        ):
            return f"option {option_number}"

    return None


def _render_legacy_property_card(prop: dict) -> None:
    passed = prop.get("passed_review", True)
    border_color = "#2ecc71" if passed else "#e74c3c"
    status_text = "Recommended" if passed else "Review flagged"

    is_available = prop.get("is_available")
    if is_available is True:
        avail_badge = "Available"
    elif is_available is False:
        avail_badge = "Unavailable"
    else:
        avail_badge = "Check site"

    nightly = prop.get("nightly_price_usd")
    price_display = f"${nightly:.0f}/night" if nightly else "Price on request"
    amenities = prop.get("amenities", [])
    amenity_tags = " ".join([f"`{a}`" for a in amenities[:6]]) if isinstance(amenities, list) else ""

    st.markdown(
        f"""
<div style="
    border: 1px solid {border_color};
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 16px;
    background: #ffffff;
    box-shadow: 0 2px 8px rgba(24, 37, 56, 0.06);
">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <h3 style="color: #172033; margin: 0;">{_escape(prop.get("name"))}</h3>
        <span style="color: #8a5a00; font-size: 1.1rem; font-weight: bold;">{_escape(price_display)}</span>
    </div>
    <p style="color: #516173; margin: 4px 0 8px 0;">
        {_escape(prop.get("city"))}, {_escape(prop.get("country"))} &nbsp;|&nbsp;
        {_escape(prop.get("property_type", "Property")).title()} &nbsp;|&nbsp;
        {_escape(status_text)} &nbsp;|&nbsp; {_escape(avail_badge)}
    </p>
    <p style="color: #516173; margin: 4px 0;">{amenity_tags}</p>
    <p style="color: #6b7887; font-size: 0.85rem; margin: 8px 0 4px 0;
        font-style: italic;">{_escape(prop.get("reviewer_notes", ""))}</p>
</div>
""",
        unsafe_allow_html=True,
    )


def render_live_availability(live_availability: dict | None) -> None:
    if not live_availability:
        return

    st.markdown("---")
    st.markdown("### Live Availability")
    message = live_availability.get("message")
    if message:
        st.info(message)

    rooms = live_availability.get("rooms") or []
    if not rooms:
        return

    for index, room in enumerate(rooms, start=1):
        price = room.get("price")
        currency = room.get("currency") or ""
        price_text = f"{price} {currency}" if price is not None else "Price not returned"
        inventory = room.get("inventory_count")
        booking_url = _clean_url(room.get("booking_url") or live_availability.get("booking_url"))

        st.markdown(
            f"""
<div style="
    border: 1px solid #c78c20;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 16px;
    background: #ffffff;
    box-shadow: 0 2px 8px rgba(24, 37, 56, 0.06);
">
    <h4 style="color: #172033; margin: 0;">{index}. {_escape(room.get("room_name") or "Room")}</h4>
    <p style="color: #516173; margin: 4px 0;">{_escape(room.get("rate_plan_name") or "Rate plan")}</p>
    <p style="color: #8a5a00; margin: 4px 0;"><strong>{_escape(price_text)}</strong></p>
    <p style="color: #516173; margin: 4px 0;">Inventory: {_escape(inventory if inventory is not None else "Not returned")}</p>
    <p style="color: #516173; margin: 4px 0;">Cancellation: {_escape(room.get("cancellation_summary") or "Not returned")}</p>
</div>
""",
            unsafe_allow_html=True,
        )
        if booking_url:
            st.link_button("Booking link", booking_url)
