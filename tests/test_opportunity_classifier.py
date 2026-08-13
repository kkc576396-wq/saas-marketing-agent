"""SmartPush opportunity-classification tests."""

from workflow.opportunity_classifier import (
    COMPETITOR_OPPORTUNITY,
    EDUCATIONAL_CONTENT,
    PRODUCT_FEEDBACK,
    classify_insight,
)


def test_klaviyo_pricing_complaint_is_competitor_opportunity():
    result = classify_insight("Klaviyo pricing complaint from Shopify merchants")

    assert result["opportunity_type"] == COMPETITOR_OPPORTUNITY


def test_email_segmentation_guide_is_educational_content():
    result = classify_insight("Email segmentation guide and best practices")

    assert result["opportunity_type"] == EDUCATIONAL_CONTENT


def test_shopify_integration_request_is_product_feedback():
    result = classify_insight("Request for Shopify integration")

    assert result["opportunity_type"] == PRODUCT_FEEDBACK
