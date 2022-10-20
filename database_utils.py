import csv
import os
import re
from typing import Tuple, Optional

with open(os.path.dirname(__file__) + r'/new_iso_lookup.csv', mode='r') as infile:
    reader = csv.reader(infile)
    reader_rows = [row for row in reader]
    ISO_DICT = {row[3]: row[5] for row in reader_rows[1:]}
    DATABASE_DICT = {row[5]: row[3] for row in reader_rows[1:]}

with open(os.path.dirname(__file__) + r'/mobile_database_lookup.csv', mode='r') as infile:
    reader = csv.reader(infile)
    reader_rows = [row for row in reader]
    MOBILE_DB_LIST = [row[1] for row in reader_rows[1:]]


def parse_database(database_code: str,
                   mobile: bool = False,
                   if_none: str = 'us') -> Optional[str]:
    if database_code is None:
        database_code = if_none
    if re.match(r"^[A-Z]{3}$", database_code):
        database_code = iso_to_database(iso_code=database_code)

    if mobile:
        mobile_code = f"mobile-{database_code}"
        if mobile_code in MOBILE_DB_LIST:
            return mobile_code
        else:
            return None
    else:
        return database_code


def iso_to_database(iso_code: str):
    if iso_code == "GBR":
        return "uk"
    elif iso_code == "USA":
        return "us"
    else:
        return ISO_DICT.get(iso_code, 'zz')


def database_to_iso(database: str) -> Tuple[str, bool]:
    _mobile = False
    if m := re.match(r'mobile-(\w+)', database):
        _mobile = True
        database = m.group(1)
    iso_code = DATABASE_DICT.get(database, 'ZZZ')
    return iso_code, _mobile
