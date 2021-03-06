import datetime
import dateutil.relativedelta as rd
import pandas as pd
import requests
import re
from typing import List, Optional, Union

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

        if database is None:
            database = self.default_database
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

        if database is None:
            database = self.default_database
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

        if database is None:
            database = self.default_database
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

        if database is None:
            database = self.default_database
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

        if database is None:
            database = self.default_database
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

        if database is None:
            database = self.default_database
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

        if database is None:
            database = self.default_database
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

        if database is None:
            database = self.default_database
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

    def batch_keyword_overview(self,
                               phrases: List[str],
                               database: str = None,
                               display_date: str = None,
                               export_columns: List[str] = None) -> Optional[pd.DataFrame]:
        # https://developer.semrush.com/api/v3/analytics/keyword-reports/#keyword-overview-one-database/

        if database is None:
            database = self.default_database
        if export_columns is None:
            export_columns = ["Ph", "Nq", "Cp", "Co", "Nr", "Td", "In"]
        if display_date is None:
            display_date = default_date()
        display_date = display_date[:-2] + "15"

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

        if database is None:
            database = self.default_database
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

        if database is None:
            database = self.default_database
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

        if database is None:
            database = self.default_database
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

        if database is None:
            database = self.default_database
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

        if database is None:
            database = self.default_database
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

        if database is None:
            database = self.default_database
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

        if database is None:
            database = self.default_database
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

        if database is None:
            database = self.default_database
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


def default_date():
    _d = datetime.datetime.today() + rd.relativedelta(months=-1)
    return _d.strftime("%Y%m%d")


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns_dict = {col: "sr_" + col.lower().replace("(%)", "pc").replace(" ", "_") for col in df.columns}
    columns_dict.update({"Keyword": "phrase"})
    return df.rename(columns=columns_dict)
