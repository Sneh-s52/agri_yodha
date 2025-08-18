import requests
import time
import logging
import datetime
from typing import Dict, List, Any
from config import MANDI_API_KEY, MANDI_API_BASE_URL


logger = logging.getLogger(__name__)


class MandiPriceAPI:
    """Client for accessing the Variety-wise Daily Market Prices API on data.gov.in"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or MANDI_API_KEY
        # Ensure this is exactly the “resource” URL from data.gov.in, not just a catalog endpoint
        self.base_url = MANDI_API_BASE_URL
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Agricultural-Market-Agent/1.0',
            'Accept': 'application/json'
        })

    def _make_request(self, params: Dict[str, Any], retries: int = MAX_RETRY_ATTEMPTS) -> List[Dict]:
        """Internal: make the API call, retry on failure, return the `records` list."""
        payload = {
            'api-key': self.api_key,
            'format': 'json',
            **params
        }

        for attempt in range(retries + 1):
            try:
                logger.info(f"API request attempt {attempt + 1}: {params}")
                resp = self.session.get(self.base_url, params=payload, timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
                body = resp.json()

                # The JSON structure has top-level metadata keys, then "records"
                records = body.get('records', [])
                logger.info(f"Retrieved {len(records)} records")
                return records

            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == retries:
                    logger.error("All retry attempts failed")
                    raise
                time.sleep(2 ** attempt)

        return []

    def fetch_latest_prices(
        self,
        state: str = None,
        district: str = None,
        commodity: str = None,
        arrival_date: str = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Fetch the latest mandi prices, filtered by State, District, and/or Commodity.

        Example:
            fetch_latest_prices(
                state="Rajasthan",
                district="Jodhpur",
                commodity="Wheat",
                limit=10
            )
        """
        params: Dict[str, Any] = {'limit': limit}
        if state:
            params['filters[State]'] = state
        if district:
            params['filters[District]'] = district
        if commodity:
            params['filters[Commodity]'] = commodity
        if arrival_date:
            params['filters[Arrival_Date]'] = arrival_date

        return self._make_request(params)

    def fetch_historical_data(
            self,
            state: str,
            district: str,
            commodity: str,
            start_date: str,
            end_date: str,
            limit: int = 1000
    ) -> List[Dict]:
        """
        Fetch historical price data by making API calls for each date in the range
        from `start_date` to `end_date`.

        Dates must be "YYYY-MM-DD".
        """
        try:
            start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.datetime.strptime(end_date, '%Y-%m-%d')
            all_records = []

            current_date = start
            while current_date <= end:
                params: Dict[str, Any] = {
                    'filters[State]': state,
                    'filters[District]': district,
                    'filters[Commodity]': commodity,
                    'filters[Arrival_Date]': current_date.strftime('%Y-%m-%d'),
                    'limit': limit
                }
                records = self._make_request(params)
                all_records.extend(records)
                current_date += datetime.timedelta(days=1)

            return all_records

        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return []

    def fetch_all_commodities(self) -> List[str]:
        """Return a sorted list of all distinct commodities."""
        try:
            records = self._make_request({'limit': 5000})
            commodities = {r['Commodity'] for r in records if r.get('Commodity')}
            return sorted(commodities)
        except Exception as e:
            logger.error(f"Error fetching commodities: {e}")
            return []

    def fetch_all_markets(self, state: str = None) -> List[str]:
        """Return a sorted list of all distinct markets (APMCs), optionally filtered by State."""
        try:
            params: Dict[str, Any] = {'limit': 5000}
            if state:
                params['filters[State]'] = state
            records = self._make_request(params)
            markets = {r['Market'] for r in records if r.get('Market')}
            return sorted(markets)
        except Exception as e:
            logger.error(f"Error fetching markets: {e}")
            return []
