import csv
import os

with open(os.path.dirname(__file__) + r'/new_iso_lookup.csv', mode='r') as infile:
    reader = csv.reader(infile)
    ISO_DICT = {rows[3]: rows[5] for rows in reader}


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
        return ISO_DICT.get(iso_code, 'zz')
