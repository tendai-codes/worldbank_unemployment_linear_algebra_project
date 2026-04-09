import requests
from time import sleep
from src.config import BASE_URL, COUNTRY_METADATA_URL, PER_PAGE, TIMEOUT


RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 5
BACKOFF_SECONDS = 1.5


def _request_json_with_retries(url, params):
    """
    Perform a GET request with exponential backoff for transient failures.
    """
    last_error = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.get(url, params=params, timeout=TIMEOUT)

            if response.status_code in RETRYABLE_STATUS_CODES:
                response.raise_for_status()

            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            retryable = status_code in RETRYABLE_STATUS_CODES or status_code is None

            if attempt == MAX_ATTEMPTS or not retryable:
                raise

            wait_seconds = BACKOFF_SECONDS * (2 ** (attempt - 1))
            print(
                f"Request failed (attempt {attempt}/{MAX_ATTEMPTS}, "
                f"status={status_code}). Retrying in {wait_seconds:.1f}s..."
            )
            sleep(wait_seconds)

    if last_error is not None:
        raise last_error


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

        payload = _request_json_with_retries(url, params)
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

        payload = _request_json_with_retries(COUNTRY_METADATA_URL, params)
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
