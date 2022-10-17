import datetime
import dateutil.relativedelta as rd
import pandas as pd
import numpy as np
import requests
import re
from typing import List, Optional, Union
from .database_utils import parse_database

cost_per_line_dict = {
    "domain_organic": 10,
    "domain_adwords": 20,
    "url_organic": 10,
    "url_adwords": 20,
    "domain_organic_organic": 40,
    "domain_adwords_adwords": 40,
    "domain_domains": 80,
    "phrase_all": 10,
    "phrase_this": 10,
    "phrase_these": 10,
    "phrase_organic": 10,
    "phrase_adwords": 20,
    "phrase_related": 40,
    "phrase_adwords_historical": 100,
    "phrase_fullsearch": 20,
    "phrase_questions": 40,
    "phrase_kdi": 50,
    "domain_rank_history": 10
}


class SemRushClient:
    def __init__(self,
                 api_key: str,
                 default_database: str = "uk"):

        self.endpoint: str = "https://api.semrush.com/"
        self.key: str = api_key
        self.default_database: str = default_database
        self._cost: int = 0

    @property
    def cost(self):
        return self._cost

    def dict_to_call(self, api_dict: dict):
        call_list = [self.endpoint, "?"]
        call_list.extend(["key=", self.key])
        for (k, v) in zip(api_dict.keys(), api_dict.values()):
            if type(v) is list:
                v = ",".join(v)
            elif type(v) is int or type(v) is float:
                v = str(v)
            call_list.extend(["&", k, "=", v])
        return ''.join(call_list)

    def make_call(self,
                  api_dict,
                  raw_text: bool = False) -> Union[pd.DataFrame, str, None]:

        if "database" in api_dict.keys():
            api_dict['database'] = parse_database(database_code=api_dict.get('database'),
                                                  if_none=self.default_database)

        api_call = self.dict_to_call(api_dict=api_dict)
        response = requests.get(url=api_call)

        if re.match(r"ERROR", response.text):
            print(response.text)
            return api_call

        if raw_text:
            return response.text

        lines = response.text.split("\r\n")
        items = [line.split(";") for line in lines]
        if len(items) > 1:
            _df = pd.DataFrame(items[1:], columns=items[0])
            _df.dropna(axis=0, how='any', inplace=True)
            _call_cost = len(_df) * cost_per_line_dict.get(api_dict.get("type"), 0)
            self._cost += _call_cost
            print(f"semrushapi_wrapper response cost {_call_cost} api credits")
            _df = clean_columns(_df)
            if 'trends' in _df.columns:
                _df['sr_trends'] = _df['sr_trends'].apply(lambda s: [float(item) for item in s.split(",")])
            return _df
        else:
            return None

    def domain_organic_search_keywords(self,
                                       domain: str,
                                       database: str = None,
                                       display_limit: int = 50,
                                       display_offset: int = 0,
                                       display_sort: str = "po_asc",
                                       display_date: str = None,
                                       export_columns: List[str] = None,
                                       display_filter: List[str] = None,
                                       display_positions: str = None) -> Optional[pd.DataFrame]:

        if export_columns is None:
            export_columns = ["Ph", "Po", "Pp", "Pd", "Nq", "Cp", "Ur",
                              "Tr", "Tc", "Co", "Nr", "Td"]
        if display_date is None:
            display_date = default_date()
        display_date = display_date[:-2] + "15"

        api_dict = {
            "type": "domain_organic",
            "domain": domain,
            "database": database,
            "display_limit": display_limit,
            "display_offset": display_offset,
            "display_sort": display_sort,
            "export_columns": export_columns,
            "display_date": display_date
        }

        if display_filter is not None:
            api_dict["display_filter"] = display_filter
        if display_positions is not None:
            api_dict["display_positions"] = display_positions

        return self.make_call(api_dict=api_dict)

    def domain_paid_search_keywords(self,
                                    domain: str,
                                    database: str = None,
                                    display_limit: int = 50,
                                    display_offset: int = 0,
                                    display_sort: str = "po_asc",
                                    display_date: str = None,
                                    export_columns: List[str] = None,
                                    display_filter: List[str] = None,
                                    display_positions: str = None) -> Optional[pd.DataFrame]:

        if export_columns is None:
            export_columns = ["Ph", "Po", "Pp", "Pd", "Ab", "Nq", "Cp",
                              "Tg", "Tr", "Tc", "Co", "Nr", "Td", "Tt",
                              "Ds", "Vu", "Ur", "Ts", "Un"]
        if display_date is None:
            display_date = default_date()
        display_date = display_date[:-2] + "15"

        api_dict = {
            "type": "domain_adwords",
            "domain": domain,
            "database": database,
            "display_limit": display_limit,
            "display_offset": display_offset,
            "display_sort": display_sort,
            "export_columns": export_columns,
            "display_date": display_date
        }

        if display_filter is not None:
            api_dict["display_filter"] = display_filter
        if display_positions is not None:
            api_dict["display_positions"] = display_positions

        return self.make_call(api_dict=api_dict)

    def url_organic_search_keywords(self,
                                    url: str,
                                    database: str = None,
                                    display_limit: int = 50,
                                    display_offset: int = 0,
                                    display_sort: str = "po_asc",
                                    display_date: str = None,
                                    export_columns: List[str] = None,
                                    display_filter: List[str] = None) -> Optional[pd.DataFrame]:

        if export_columns is None:
            export_columns = ["Ph", "Po", "Pp", "Nq", "Cp", "Co", "Kd",
                              "Tr", "Tg", "Tc", "Nr", "Td", "Fp", "Fk",
                              "Ts"]
        if display_date is None:
            display_date = default_date()
        display_date = display_date[:-2] + "15"

        api_dict = {
            "type": "url_organic",
            "url": url,
            "database": database,
            "display_limit": display_limit,
            "display_offset": display_offset,
            "display_sort": display_sort,
            "export_columns": export_columns,
            "display_date": display_date
        }

        if display_filter is not None:
            api_dict["display_filter"] = display_filter

        return self.make_call(api_dict=api_dict)

    def url_paid_search_keywords(self,
                                 url: str,
                                 database: str = None,
                                 display_limit: int = 50,
                                 display_offset: int = 0,
                                 display_sort: str = "po_asc",
                                 display_date: str = None,
                                 export_columns: List[str] = None,
                                 display_filter: List[str] = None) -> Optional[pd.DataFrame]:

        if export_columns is None:
            export_columns = ["Ph", "Po", "Nq", "Cp", "Co", "Tg", "Tr",
                              "Tc", "Nr", "Td", "Tt", "Ds", "Ts"]
        if display_date is None:
            display_date = default_date()
        display_date = display_date[:-2] + "15"

        api_dict = {
            "type": "url_adwords",
            "url": url,
            "database": database,
            "display_limit": display_limit,
            "display_offset": display_offset,
            "display_sort": display_sort,
            "export_columns": export_columns,
            "display_date": display_date
        }

        if display_filter is not None:
            api_dict["display_filter"] = display_filter

        return self.make_call(api_dict=api_dict)

    def organic_competitors(self,
                            domain: str,
                            database: str = None,
                            display_limit: int = 10,
                            display_offset: int = 0,
                            display_sort: str = "np_desc",
                            display_date: str = None,
                            export_columns: List[str] = None) -> Optional[pd.DataFrame]:

        if export_columns is None:
            export_columns = ["Dn", "Cr", "Np", "Or", "Ot", "Oc", "Ad"]
        if display_date is None:
            display_date = default_date()
        display_date = display_date[:-2] + "15"

        api_dict = {
            "type": "domain_organic_organic",
            "domain": domain,
            "database": database,
            "display_limit": display_limit,
            "display_offset": display_offset,
            "display_sort": display_sort,
            "export_columns": export_columns,
            "display_date": display_date
        }

        return self.make_call(api_dict=api_dict)

    def paid_competitors(self,
                         domain: str,
                         database: str = None,
                         display_limit: int = 10,
                         display_offset: int = 0,
                         display_sort: str = "np_desc",
                         display_date: str = None,
                         export_columns: List[str] = None) -> Optional[pd.DataFrame]:

        if export_columns is None:
            export_columns = ["Dn", "Cr", "Np", "Ad", "At", "Ac", "Or"]
        if display_date is None:
            display_date = default_date()
        display_date = display_date[:-2] + "15"

        api_dict = {
            "type": "domain_adwords_adwords",
            "domain": domain,
            "database": database,
            "display_limit": display_limit,
            "display_offset": display_offset,
            "display_sort": display_sort,
            "export_columns": export_columns,
            "display_date": display_date
        }

        return self.make_call(api_dict=api_dict)

    def domain_vs_domain(self,
                         domains: List[str],
                         _sign: str = "*",
                         _type: str = "or",
                         database: str = None,
                         display_limit: int = 10,
                         display_offset: int = 0,
                         display_sort: str = "p0_asc",
                         display_date: str = None,
                         export_columns: List[str] = None,
                         display_filter: str = None) -> Optional[pd.DataFrame]:

        if export_columns is None:
            export_columns = ["Ph", "P0", "P1", "P2", "P3", "P4", "Nr", "Cp", "Nq", "Kd", "Co", "Td"]
        if display_date is None:
            display_date = default_date()
        display_date = display_date[:-2] + "15"

        api_dict = {
            "type": "domain_domains",
            "domains": "|".join([f"{_sign}|{_type}|{_d}" for _d in domains]),
            "database": database,
            "display_limit": display_limit,
            "display_offset": display_offset,
            "display_sort": display_sort,
            "export_columns": export_columns,
            "display_date": display_date
        }

        if display_filter is not None:
            api_dict["display_filter"] = display_filter

        return self.make_call(api_dict=api_dict)

    def keyword_overview(self,
                         phrase: str,
                         database: str = None,
                         display_date: str = None,
                         export_columns: List[str] = None) -> Optional[pd.DataFrame]:
        # https://developer.semrush.com/api/v3/analytics/keyword-reports/#keyword-overview-one-database/

        if export_columns is None:
            export_columns = ["Ph", "Nq", "Cp", "Co", "Nr", "Td", "In"]
        if display_date is None:
            display_date = default_date()
        display_date = display_date[:-2] + "15"

        api_dict = {
            "type": "phrase_this",
            "phrase": phrase,
            "database": database,
            "export_columns": export_columns,
            "display_date": display_date
        }

        return self.make_call(api_dict=api_dict)

    def big_batch_keyword_overview(self,
                                   phrases: List[str],
                                   database: str = None,
                                   display_date: str = None,
                                   export_columns: List[str] = None,
                                   partition_size: int = 100) -> Optional[pd.DataFrame]:

        phrases_partitions = [phrases[i:i + partition_size] for i in range(0, len(phrases), partition_size)]

        frames = []
        for _part in phrases_partitions:
            frames.append(
                self.batch_keyword_overview(phrases=_part,
                                            database=database,
                                            display_date=display_date,
                                            export_columns=export_columns)
            )

        return pd.concat(frames)

    def batch_keyword_overview(self,
                               phrases: List[str],
                               database: str = None,
                               display_date: str = None,
                               export_columns: List[str] = None) -> Optional[pd.DataFrame]:
        # https://developer.semrush.com/api/v3/analytics/keyword-reports/#keyword-overview-one-database/

        if export_columns is None:
            export_columns = ["Ph", "Nq", "Cp", "Co", "Nr", "Td", "In"]
        if display_date is None:
            display_date = default_date()
        display_date = display_date[:-2] + "15"

        if len(phrases) > 100:
            return self.big_batch_keyword_overview(phrases=phrases,
                                                   database=database,
                                                   display_date=display_date,
                                                   export_columns=export_columns,
                                                   partition_size=100)

        api_dict = {
            "type": "phrase_these",
            "phrase": ";".join(phrases),
            "database": database,
            "export_columns": export_columns,
            "display_date": display_date
        }

        return self.make_call(api_dict=api_dict)

    def organic_results(self,
                        phrase: str,
                        database: str = None,
                        display_date: str = None,
                        display_limit: int = 100,
                        export_columns: List[str] = None) -> Optional[pd.DataFrame]:
        # https://developer.semrush.com/api/v3/analytics/keyword-reports/#keyword-overview-one-database/

        if export_columns is None:
            export_columns = ["Dn", "Ur", "Fk", "Fp"]
        if display_date is None:
            display_date = default_date()
        display_date = display_date[:-2] + "15"

        api_dict = {
            "type": "phrase_organic",
            "phrase": phrase,
            "database": database,
            "display_limit": display_limit,
            "export_columns": export_columns,
            "display_date": display_date
        }

        return self.make_call(api_dict=api_dict)

    def paid_results(self,
                     phrase: str,
                     database: str = None,
                     display_date: str = None,
                     display_limit: int = 100,
                     export_columns: List[str] = None) -> Optional[pd.DataFrame]:
        # https://developer.semrush.com/api/v3/analytics/keyword-reports/#keyword-overview-one-database/

        if export_columns is None:
            export_columns = ["Dn", "Ur", "Vu"]
        if display_date is None:
            display_date = default_date()
        display_date = display_date[:-2] + "15"

        api_dict = {
            "type": "phrase_adwords",
            "phrase": phrase,
            "database": database,
            "display_limit": display_limit,
            "export_columns": export_columns,
            "display_date": display_date
        }

        return self.make_call(api_dict=api_dict)

    def related_keywords(self,
                         phrase: str,
                         database: str = None,
                         display_limit: int = 50,
                         display_offset: int = 0,
                         display_sort: str = "nq_asc",
                         display_date: str = None,
                         export_columns: List[str] = None,
                         display_filter: List[str] = None,
                         display_positions: str = None) -> Optional[pd.DataFrame]:

        if export_columns is None:
            export_columns = ["Ph", "Nq", "Cp", "Co", "Nr", "Td", "Rr",
                              "Fk", "In"]
        if display_date is None:
            display_date = default_date()
        display_date = display_date[:-2] + "15"

        api_dict = {
            "type": "phrase_related",
            "phrase": phrase,
            "database": database,
            "display_limit": display_limit,
            "display_offset": display_offset,
            "display_sort": display_sort,
            "export_columns": export_columns,
            "display_date": display_date
        }

        if display_filter is not None:
            api_dict["display_filter"] = display_filter
        if display_positions is not None:
            api_dict["display_positions"] = display_positions

        return self.make_call(api_dict=api_dict)

    def keyword_ads_history(self,
                            phrase: str,
                            database: str = None,
                            display_limit: int = 50,
                            display_offset: int = 0,
                            export_columns: List[str] = None) -> Optional[pd.DataFrame]:

        if export_columns is None:
            export_columns = ["Dn", "Dt", "Po", "Ur", "Tt", "Ds", "Vu", "At", "Ac", "Ad"]

        api_dict = {
            "type": "phrase_adwords_historical",
            "phrase": phrase,
            "database": database,
            "display_limit": display_limit,
            "display_offset": display_offset,
            "export_columns": export_columns,
        }

        return self.make_call(api_dict=api_dict)

    def broad_match_keyword(self,
                            phrase: str,
                            database: str = None,
                            display_limit: int = 50,
                            display_offset: int = 0,
                            display_sort: str = "nq_asc",
                            export_columns: List[str] = None,
                            display_filter: List[str] = None) -> Optional[pd.DataFrame]:

        if export_columns is None:
            export_columns = ["Ph", "Nq", "Cp", "Co", "Nr", "Td", "Fk", "In"]

        api_dict = {
            "type": "phrase_fullsearch",
            "phrase": phrase,
            "database": database,
            "display_limit": display_limit,
            "display_offset": display_offset,
            "display_sort": display_sort,
            "export_columns": export_columns
        }

        if display_filter is not None:
            api_dict["display_filter"] = display_filter

        return self.make_call(api_dict=api_dict)

    def phrase_questions(self,
                         phrase: str,
                         database: str = None,
                         display_limit: int = 50,
                         display_offset: int = 0,
                         display_sort: str = "nq_asc",
                         export_columns: List[str] = None,
                         display_filter: List[str] = None) -> Optional[pd.DataFrame]:

        if export_columns is None:
            export_columns = ["Ph", "Nq", "Cp", "Co", "Nr", "Td", "In"]

        api_dict = {
            "type": "phrase_questions",
            "phrase": phrase,
            "database": database,
            "display_limit": display_limit,
            "display_offset": display_offset,
            "display_sort": display_sort,
            "export_columns": export_columns
        }

        if display_filter is not None:
            api_dict["display_filter"] = display_filter

        return self.make_call(api_dict=api_dict)

    def keyword_difficulty(self,
                           phrase: str,
                           database: str = None,
                           export_columns: List[str] = None) -> str:

        if export_columns is None:
            export_columns = ["Ph", "Kd"]

        api_dict = {
            "type": "phrase_kdi",
            "phrase": phrase,
            "database": database,
            "export_columns": export_columns
        }

        return self.make_call(api_dict=api_dict, raw_text=True)

    def domain_overview_history(self,
                                domain: str,
                                database: str = None,
                                display_limit: int = 50,
                                display_offset: int = 0,
                                display_sort: str = "dt_desc",
                                export_columns: List[str] = None) -> Optional[pd.DataFrame]:

        if export_columns is None:
            export_columns = ["Rk", "Or", "Xn", "Ot", "Oc", "Ad", "At", "Ac", "Dt", "FKn", "FPn"]

        api_dict = {
            "type": "domain_rank_history",
            "domain": domain,
            "database": database,
            "display_limit": display_limit,
            "display_offset": display_offset,
            "display_sort": display_sort,
            "export_columns": export_columns,
        }

        return self.make_call(api_dict=api_dict)

    def multi_domain_comparison(self,
                                domain_list: List[str],
                                database: str = None,
                                export_columns: List[str] = None) -> Optional[pd.DataFrame]:

        frames = []
        for domain in domain_list:
            domain_overview_df = self.domain_overview_history(domain=domain,
                                                              database=database,
                                                              display_limit=13,
                                                              display_offset=0,
                                                              display_sort="dt_desc",
                                                              export_columns=export_columns)
            frames.append((domain, domain_overview_df))

        new_dicts = []
        for _domain, _frame in frames:
            _d = {'domain': _domain}
            _columns = _frame.columns
            for _col in _columns:
                column = _frame[_col].astype(float)
                for _month in (0, 1, 12):
                    _d[_col + '_' + str(_month)] = column[_month]
                _d[_col + '_mom'] = column[0] - column[1]
                _d[_col + '_yoy'] = column[0] - column[12]
                _d[_col + '_3month_trend'] = polyfit_trend(column[0:3])
                _d[_col + '_12month_trend'] = polyfit_trend(column[0:12])

            new_dicts.append(_d)

        return pd.DataFrame(new_dicts)


def polyfit_trend(data):
    p = np.polyfit(np.arange(0, len(data)), data.to_numpy()[::-1], deg=1)
    return p[0]


def default_date():
    _d = datetime.datetime.today() + rd.relativedelta(months=-1)
    return _d.strftime("%Y%m%d")


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns_dict = {col: "sr_" + col.lower().replace("(%)", "pc").replace(" ", "_") for col in df.columns}
    columns_dict.update({"Keyword": "phrase"})
    return df.rename(columns=columns_dict)
