BASE_URL = "https://api.worldbank.org/v2/country/all/indicator/{indicator}"
COUNTRY_METADATA_URL = "https://api.worldbank.org/v2/country/all"

START_YEAR = 2010
END_YEAR = 2025
PER_PAGE = 20000
TIMEOUT = 60
AGGREGATE_REGION_LABEL = "Aggregates"

SAVE_CSV_SNAPSHOTS = True

INDICATORS = {
    "unemployment": "SL.UEM.TOTL.ZS",
    "inflation": "FP.CPI.TOTL.ZG",
    "gdp_growth": "NY.GDP.MKTP.KD.ZG",
    "life_expectancy": "SP.DYN.LE00.IN",
    "population_growth": "SP.POP.GROW",
}