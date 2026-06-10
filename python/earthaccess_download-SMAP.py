import earthaccess

# 1. Authenticate (will prompt for your Earthdata username and password)
auth = earthaccess.login()

# 2. Search for SMAP data using a shortname and temporal/spatial bounds
results = earthaccess.search_data(
    short_name   = "SPL3SMP_E",
    version      = "006",
    bounding_box = (-79.1721, 37.3296, -77.6873, 38.4755),
    temporal     = ("2023-01-01", "2023-12-31")
)
#temporal     = ("2015-04-01", "2015-12-31")
#temporal     = ("2016-01-01", "2016-12-31")
#temporal     = ("2017-01-01", "2017-12-31")
#temporal     = ("2018-01-01", "2018-12-31")
#temporal     = ("2019-01-01", "2019-12-31")
#temporal     = ("2020-01-01", "2020-12-31")
#temporal     = ("2021-01-01", "2021-12-31")
#temporal     = ("2022-01-01", "2022-12-31")
print(f"Granules found: {len(results)}")

files = earthaccess.download(
    results,
    local_path = "/scratch/thq3hn/smap_h5/"
)

