import csv
import os


def parse_database(database_code: str,
                   if_none: str = 'us'):
    if database_code is None:
        return if_none
    if len(database_code) == 3:
        return iso_to_domain(iso_code=database_code.upper())
    else:
        return database_code.lower()


def iso_to_domain(iso_code: str):
    if iso_code == "GBR":
        return "uk"
    elif iso_code == "USA":
        return "us"
    else:
        with open(os.path.dirname(__file__) + r'/country_iso_codes.csv', mode='r') as infile:
            reader = csv.reader(infile)
            isodict = {rows[2]: rows[4] for rows in reader}
        return isodict.get(iso_code, 'zz')
