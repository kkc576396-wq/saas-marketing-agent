"""Competitor entity-importance tests."""

from workflow.entity_relevance import analyze_entity_importance, entity_roles


def _record(records, competitor):
    return next(item for item in records if item["competitor"] == competitor)


def test_title_competitor_is_primary_entity():
    records = analyze_entity_importance(
        "Klaviyo pricing complaints",
        "A Shopify merchant says list growth made the platform expensive.",
    )

    result = _record(records, "klaviyo")
    assert result["entity_role"] == "primary"
    assert result["basis"] == "title_subject"


def test_repeated_competitor_discussion_is_primary_entity():
    records = analyze_entity_importance(
        "Email platform migration",
        (
            "The team currently uses Omnisend for ecommerce campaigns. "
            "Omnisend became difficult to manage as segmentation expanded."
        ),
    )

    assert _record(records, "omnisend")["entity_role"] == "primary"


def test_single_competitor_in_vendor_list_is_incidental():
    records = analyze_entity_importance(
        "How we built our CRM stack",
        (
            "We connected customer records across the sales organization. "
            "The tools considered included Twilio, HubSpot, Klaviyo, "
            "Pipedrive, and several internal services."
        ),
    )

    assert _record(records, "klaviyo")["entity_role"] == "incidental"


def test_ambiguous_drip_coffee_is_not_an_entity():
    roles = entity_roles("Drip coffee equipment and brewing temperature")

    assert roles == {"primary": [], "supporting": [], "incidental": []}
