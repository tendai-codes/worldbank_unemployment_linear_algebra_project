import requests
from src.config import BASE_URL, COUNTRY_METADATA_URL, PER_PAGE, TIMEOUT


def fetch_indicator_data(indicator_code, start_year, end_year):
    """
    Fetch one World Bank indicator for all countries over a year range.
    """
    url = BASE_URL.format(indicator=indicator_code)
    page = 1
    all_rows = []

    while True:
        params = {
            "format": "json",
            "date": f"{start_year}:{end_year}",
            "per_page": PER_PAGE,
            "page": page,
        }

        response = requests.get(url, params=params, timeout=TIMEOUT)
        response.raise_for_status()

        payload = response.json()
        metadata = payload[0]
        rows = payload[1]

        if not rows:
            break

        all_rows.extend(rows)

        if page >= metadata["pages"]:
            break

        page += 1

    print(f"Fetched {len(all_rows)} raw rows for {indicator_code}")
    return all_rows


def fetch_country_metadata():
    """
    Fetch World Bank country metadata, including region information.
    """
    page = 1
    all_rows = []

    while True:
        params = {
            "format": "json",
            "per_page": PER_PAGE,
            "page": page,
        }

        response = requests.get(COUNTRY_METADATA_URL, params=params, timeout=TIMEOUT)
        response.raise_for_status()

        payload = response.json()
        metadata = payload[0]
        rows = payload[1]

        if not rows:
            break

        all_rows.extend(rows)

        if page >= metadata["pages"]:
            break

        page += 1

    print(f"Fetched {len(all_rows)} country metadata rows")
    return all_rows