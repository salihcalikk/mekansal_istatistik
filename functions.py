import numpy as np
from scipy.stats import chi2, mannwhitneyu
import re
import pandas as pd
from statistics import mean
import collections


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
    def __init__(self, array, q):
        self.array = array
        self.q = q

    def sort_array(self, sort_field):
        df = pd.DataFrame(self.array)
        return df.sort_values(by=[sort_field], ascending=True)

    @staticmethod
    def get_repetitive_values(sorted_list):
        repetitive_list = []
        counter = 1
        final_df = sorted_list
        for index, row in final_df.iterrows():
            if index != len(final_df) - 1 and row["count"] == final_df.iat[index + 1, 1]:
                counter += 1
                if counter == 2:
                    repetitive_list.append({"count": row["count"], "index": [index, index + 1],
                                       "new_values": [row["count"], row["count"] + 1]})
                else:
                    repetitive_list[len(repetitive_list) - 1]["index"].append(index + 1)
                    repetitive_list[len(repetitive_list) - 1]["new_values"].append(repetitive_list[len(repetitive_list) - 1]["count"] + counter - 1)
            else:
                counter = 1
            for key, i in enumerate(repetitive_list):
                i["mean_value"] = mean(i["new_values"])
        return [repetitive_list, final_df]

    def ordered_new_values(self, sort_field):
        sorted_list = self.sort_array(sort_field)
        repetitive_list, ordered_list = self.get_repetitive_values(sorted_list)
        for i in repetitive_list:
            ordered_list.loc[i["index"][0]:i["index"][len(i["index"]) - 1], [sort_field]] = i["mean_value"]
        return ordered_list

    @staticmethod
    def seperate_list(ordered_list, sort_field):
        lists = []
        gb = ordered_list.groupby(sort_field)
        for x in gb.groups:
            lists.append(gb.get_group(x))
        return lists

    @staticmethod
    def u_test_score(sample_list, sum_column):
        score_list = []
        n_list = []
        r_list = []
        sum_column_index = sample_list[0].columns.get_loc(sum_column)
        for i in sample_list:
            row_len, column_len = i.shape
            sum_column = i.sum(axis=0)[sum_column_index]
            n_list.append(row_len)
            r_list.append(sum_column)
        for i in range(len(sample_list)):
            score_list.append(np.prod(n_list) + ((n_list[i] * (n_list[i] + 1)) / 2) - r_list[i])
        return min(score_list)

    @staticmethod
    def get_p_value(sample_list):
        U1, p = mannwhitneyu(sample_list[0]["count"], sample_list[1]["count"])
        return p

    def result(self, count_column, sample_column):
        ordered_list = self.ordered_new_values(count_column)
        sample_list = self.seperate_list(ordered_list, sample_column)
        u_value = self.u_test_score(sample_list, count_column)
        p_value = self.get_p_value(sample_list)
        result_text = ""
        result = {"q": f"""%{self.q}""", "p_value": p_value,
                "u_value": u_value}
        if p_value < (1 - (self.q / 100)):
            result_text = f"""Yokluk Hipotezi Reddedilmiştir. Örneklemin %{self.q}
             güven düzeyinde farklı olduğu sonucuna varılmıştır. İstatistiksel açıdan anlamlı bulunmuştur."""
            result['description'] = re.sub(' +', ' ', result_text).replace('\n', '')
        else:
            result_text = f"""Yokluk Hipotezi Reddedilememiştir. Örneklemin %{self.q}
             güven düzeyinde farklı olmadığı sonucuna varılmıştır."""
            result['description'] = re.sub(' +', ' ', result_text).replace('\n', '')
        return result


class KruskalWallisHTest:
    def __init__(self, array, q):
        self.array = array
        self.q = q

    def sort_array(self, sort_field):
        df = pd.DataFrame(self.array)
        return df.sort_values(by=[sort_field], ascending=True)

    @staticmethod
    def get_repetitive_values(sorted_list):
        repetitive_list = []
        counter = 1
        final_df = sorted_list
        for index, row in final_df.iterrows():
            if index != len(final_df) - 1 and row["count"] == final_df.iat[index + 1, 1]:
                counter += 1
                if counter == 2:
                    repetitive_list.append({"count": row["count"], "index": [index, index + 1],
                                       "new_values": [row["count"], row["count"] + 1]})
                else:
                    repetitive_list[len(repetitive_list) - 1]["index"].append(index + 1)
                    repetitive_list[len(repetitive_list) - 1]["new_values"].append(repetitive_list[len(repetitive_list) - 1]["count"] + counter - 1)
            else:
                counter = 1
            for key, i in enumerate(repetitive_list):
                i["mean_value"] = mean(i["new_values"])
        return [repetitive_list, final_df]

    def ordered_new_values(self, sort_field):
        sorted_list = self.sort_array(sort_field)
        repetitive_list, ordered_list = self.get_repetitive_values(sorted_list)
        for i in repetitive_list:
            ordered_list.loc[i["index"][0]:i["index"][len(i["index"]) - 1], [sort_field]] = i["mean_value"]
        return ordered_list

    @staticmethod
    def seperate_list(ordered_list, sort_field):
        lists = []
        gb = ordered_list.groupby(sort_field)
        for x in gb.groups:
            lists.append(gb.get_group(x))
        return lists

    def h_test_score(self, sample_list, sum_column):
        count = 0
        total_row_len = 0
        row_len_list = []
        sum_column_index = sample_list[0].columns.get_loc(sum_column)
        for i in sample_list:
            row_len, column_len = i.shape
            row_len_list.append(row_len)
            sum_column = i.sum(axis=0)[sum_column_index]
            count += ((sum_column**2) / row_len)
            total_row_len += row_len
        result = ((12 / (total_row_len * (total_row_len + 1))) * (count)) - (3 * (total_row_len + 1))
        if [item for item, count in collections.Counter(row_len_list).items() if count > 1]:
            sorted_list = self.sort_array("count")
            repetative = self.get_repetitive_values(sorted_list)
            t = 0
            for i in repetative[0]:
                len_index = len(i["index"])
                t += (pow(len_index, 3) - len_index)
            result = result / (1 - (t / (pow(total_row_len, 3) - total_row_len)))
        return result

    def get_alpha(self, sample_list):
        q = self.q / 100
        df = (len(sample_list) - 1)
        result = chi2.ppf(q, df=df)
        return {"q": q, "df": df, "result": result}

    def result(self, count_column, sample_column):
        ordered_list = self.ordered_new_values(count_column)
        sample_list = self.seperate_list(ordered_list, sample_column)
        h_test = self.h_test_score(sample_list, count_column)
        alpha = self.get_alpha(sample_list)
        result_text = ""
        result = {"q": f"""%{int(alpha["q"] * 100)}""", "df": alpha["df"],
                "alpha": alpha["result"], "h-test": h_test}
        if h_test >= alpha["result"]:
            result_text = f"""Yokluk Hipotezi Reddedilmiştir. Örneklemin %{int(alpha["q"] * 100)}
             güven düzeyinde farklı olduğu sonucuna varılmıştır. İstatistiksel açıdan anlamlı bulunmuştur."""

            result['description'] = re.sub(' +', ' ', result_text).replace('\n', '')
        else:
            result_text = f"""Yokluk Hipotezi Reddedilememiştir. Örneklemin %{alpha["q"] * 100}
             güven düzeyinde farklı olmadığı sonucuna varılmıştır."""
            result['description'] = re.sub(' +', ' ', result_text).replace('\n', '')
        return result





        # result = {"q": f"""%{self.q}""", "p_value": p_value,
        #           "u_value": u_value}
        # if p_value < self.q:
        #     result_text = f"""Yokluk Hipotezi Reddedilmiştir. Örneklemin %{self.q}
        #              güven düzeyinde farklı olduğu sonucuna varılmıştır. İstatistiksel açıdan anlamlı bulunmuştur."""
        #     result['description'] = re.sub(' +', ' ', result_text).replace('\n', '')
        # else:
        #     result_text = f"""Yokluk Hipotezi Reddedilememiştir. Örneklemin %{self.q}
        #              güven düzeyinde farklı olmadığı sonucuna varılmıştır."""
        #     result['description'] = re.sub(' +', ' ', result_text).replace('\n', '')
        # return result