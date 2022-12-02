import numpy as np
from scipy.stats import chi2
import re


class ChiKare:
    def __init__(self, array, q):
        self.array = array
        self.q = q

    def get_array_values_expected(self, row, column):
        sum_column = self.array.sum(axis=0)[column]
        sum_row = self.array.sum(axis=1)[row]
        sum_whole = self.array.sum()
        result = (sum_row * sum_column) / sum_whole
        return result

    def get_array_values(self):
        result = []
        for key_row, i in enumerate(self.array):
            for key_column, j in enumerate(i):
                result.append({"row": key_row + 1, "column": key_column + 1, "data": j,
                               "expected": self.get_array_values_expected(key_row, key_column)})
        return result

    def get_serbestlik(self):
        row_len, column_len = self.array.shape
        result = (row_len - 1) * (column_len - 1)
        return result

    def get_alpha(self):
        q = self.q / 100
        df = self.get_serbestlik()
        result = chi2.ppf(q, df=df)
        return {"q": q, "df": df, "result": result}

    def get_chi_kare(self):
        row_len, column_len = self.array.shape
        result = 0
        if row_len == 2 and column_len == 2:
            n = np.sum(self.array)
            a = self.array[0, 0]
            b = self.array[0, 1]
            c = self.array[1, 0]
            d = self.array[1, 1]
            result = (n * ((abs((a * d) - (b * c)) - (n / 2))) ** 2) / ((a + b) * (c + d) * (a + c) * (b + d))
        else:
            array_value = self.get_array_values()
            for i in array_value:
                append_data = ((i["data"] - i["expected"]) ** 2) / i["expected"]
                result += append_data
        return result

    def chi_kare_result(self):
        chi_kare = self.get_chi_kare()
        alpha = self.get_alpha()
        result_text = ""
        result = {"q": f"""%{int(alpha["q"] * 100)}""", "df": alpha["df"],
                "alpha": alpha["result"], "chi-kare": chi_kare}
        if chi_kare >= alpha["result"]:
            result_text = f"""Yokluk Hipotezi Reddedilmiştir. Örneklemin %{int(alpha["q"] * 100)}
             güven düzeyinde farklı olduğu sonucuna varılmıştır. İstatistiksel açıdan anlamlı bulunmuştur."""

            result['description'] = re.sub(' +', ' ', result_text).replace('\n', '')
        else:
            result_text = f"""Yokluk Hipotezi Reddedilememiştir. Örneklemin %{alpha["q"] * 100}
             güven düzeyinde farklı olmadığı sonucuna varılmıştır."""
            result['description'] = re.sub(' +', ' ', result_text).replace('\n', '')
        return result


class MannWhitneyUtest:
    def __init__(self, a):
        self.a = a


class KruskalWallisHTest:
    def __init__(self, a):
        self.a = a