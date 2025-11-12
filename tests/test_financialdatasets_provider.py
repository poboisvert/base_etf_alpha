from datetime import datetime

import httpx
import pytest

from portwine.data.providers.financialdatasets import FinancialDatasetsProvider


def test_financialdatasets_provider_normalizes_prices():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-API-KEY"] == "test-key"
        assert request.url.path == "/prices/"
        assert request.url.params["ticker"] == "AAPL"
        assert request.url.params["interval"] == "day"

        response_payload = {
            "prices": [
                {
                    "open": 228,
                    "close": 228.1,
                    "high": 228.1,
                    "low": 227.71,
                    "volume": 8199,
                    "time": "2024-10-14T04:00:00Z",
                },
                {
                    "open": 228.11,
                    "close": 228.15,
                    "high": 228.25,
                    "low": 228,
                    "volume": 2319,
                    "time": "2024-10-15 04:01:00 EDT",
                },
            ]
        }
        return httpx.Response(200, json=response_payload)

    client = httpx.Client(base_url="https://api.financialdatasets.ai", transport=httpx.MockTransport(handler))
    provider = FinancialDatasetsProvider("test-key", client=client)

    data = provider.get_data("AAPL", datetime(2024, 10, 14))

    assert datetime(2024, 10, 14, 4, 0, 0) in data
    assert data[datetime(2024, 10, 14, 4, 0, 0)]["close"] == pytest.approx(228.1)

    assert datetime(2024, 10, 15, 4, 1, 0) in data
    assert data[datetime(2024, 10, 15, 4, 1, 0)]["volume"] == pytest.approx(2319)


def test_financialdatasets_provider_handles_pagination():
    call_count = {"value": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["value"] += 1
        if call_count["value"] == 1:
            payload = {
                "prices": [
                    {
                        "open": 100,
                        "high": 110,
                        "low": 95,
                        "close": 105,
                        "volume": 5000,
                        "time": "2024-01-01T00:00:00Z",
                    }
                ],
                "next_page_url": "https://api.financialdatasets.ai/prices/?ticker=AAPL&page=2",
            }
        else:
            payload = {
                "prices": [
                    {
                        "open": 106,
                        "high": 112,
                        "low": 101,
                        "close": 108,
                        "volume": 4500,
                        "time": "2024-01-02T00:00:00Z",
                    }
                ]
            }
        return httpx.Response(200, json=payload)

    client = httpx.Client(base_url="https://api.financialdatasets.ai", transport=httpx.MockTransport(handler))
    provider = FinancialDatasetsProvider("test-key", client=client)

    data = provider.get_data("AAPL", datetime(2024, 1, 1))

    assert len(data) == 2
    assert datetime(2024, 1, 1, 0, 0) in data
    assert datetime(2024, 1, 2, 0, 0) in data

