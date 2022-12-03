from functions import ChiKare, MannWhitneyUtest, KruskalWallisHTest
import numpy as np

test_arrays = [{"method": "chi-kare", "q": 95, "array": np.array([[34, 63], [97, 16]])},
               {"method": "chi-kare", "q": 95, "array": np.array([[14, 43, 27, 45, 32], [16, 21, 34, 61, 15]])},
               {"method": "chi-kare", "q": 95, "array": np.array([[17, 2, 1, 1], [14, 3, 2, 1],
                                            [10, 1, 0, 0], [13, 2, 3, 2],
                                            [19, 3, 2, 0]])},
               {"method": "mann-whitney-u-test", "q": 95,
                "array": [
                    {"park_name": "Özgürlük Parkı", "count": 1, "borough": "A"},
                    {"park_name": "Hastane Parkı", "count": 2, "borough": "B"},
                    {"park_name": "Aslanlı Park", "count": 3, "borough": "A"},
                    {"park_name": "Kral Mezarı Parkı", "count": 4, "borough": "B"},
                    {"park_name": "Bayram Parkı", "count": 5, "borough": "A"},
                    {"park_name": "Ayrancılar Anıtı", "count": 6, "borough": "A"},
                    {"park_name": "Deniz Park", "count": 6, "borough": "A"},
                    {"park_name": "Kuşluk Parkı", "count": 8, "borough": "B"},
                    {"park_name": "Hatıra Ormanı", "count": 9, "borough": "A"},
                    {"park_name": "Şahin Tepesi", "count": 10, "borough": "A"},
                    {"park_name": "Gençlik Parkı", "count": 11, "borough": "B"},
                    {"park_name": "Zafer Anıtı", "count": 12, "borough": "B"},
                    {"park_name": "Sevgi Parkı", "count": 12, "borough": "A"},
                    {"park_name": "İkinci Bahar Parkı", "count": 12, "borough": "B"},
                    {"park_name": "Kurtuluş Parkı", "count": 15, "borough": "B"},
                    {"park_name": "Sahil Parkı", "count": 16, "borough": "A"},
                    {"park_name": "Büfe Park", "count": 17, "borough": "B"},
                ]},
               {"method": "kruskal-wallis-h-test", "q": 95,
                "array": [
                    {"construction_name": "Han Konağı", "count": 1, "borough": "A"},
                    {"construction_name": "Ağalar Köşkü", "count": 2, "borough": "B"},
                    {"construction_name": "Hanım Kökü", "count": 3, "borough": "D"},
                    {"construction_name": "Aslanlı Kale", "count": 3, "borough": "A"},
                    {"construction_name": "Beyaz Kule", "count": 3, "borough": "A"},
                    {"construction_name": "Ağaçlı Çarşı", "count": 6, "borough": "A"},
                    {"construction_name": "Aynalı Konak", "count": 7, "borough": "A"},
                    {"construction_name": "Kapısız Kule", "count": 8, "borough": "B"},
                    {"construction_name": "Sazlık Hanı", "count": 9, "borough": "A"},
                    {"construction_name": "Saray Köşkü", "count": 10, "borough": "C"},
                    {"construction_name": "Sıralı İkiz Konak", "count": 11, "borough": "A"},
                    {"construction_name": "Züğürtler Köşkü", "count": 12, "borough": "B"},
                    {"construction_name": "Sarmalı Konak", "count": 13, "borough": "B"},
                    {"construction_name": "Yiğit Kulesi", "count": 14, "borough": "B"},
                    {"construction_name": "Sırmalı Köşk", "count": 15, "borough": "D"},
                    {"construction_name": "Mavi Saray", "count": 16, "borough": "C"},
                    {"construction_name": "Arap Köşkü", "count": 17, "borough": "C"},
                    {"construction_name": "Kapısız Köşk", "count": 18, "borough": "D"},
                    {"construction_name": "Dere Han", "count": 19, "borough": "B"},
                    {"construction_name": "Kuleli Çarşı", "count": 19, "borough": "A"},
                    {"construction_name": "Laleli Köşk", "count": 21, "borough": "C"},
                    {"construction_name": "Güllü Han", "count": 22, "borough": "C"},
                    {"construction_name": "Sıvasız Kule", "count": 23, "borough": "D"},
                    {"construction_name": "Sağır Han", "count": 24, "borough": "C"},
                    {"construction_name": "Kuş Kulesi", "count": 24, "borough": "D"},
                    {"construction_name": "Yazmalı Köşk", "count": 26, "borough": "D"},
                    {"construction_name": "Edalı Konak", "count": 27, "borough": "C"},
                ]}
               ]

result_list = []
for i in test_arrays:
    if i["method"] == "chi-kare":
        chi_kare = ChiKare(array=i["array"], q=i["q"])
        result = chi_kare.chi_kare_result()
        result_list.append(result)
    elif i["method"] == "mann-whitney-u-test":
        mann_whitney = MannWhitneyUtest(array=i["array"], q=i["q"])
        result = mann_whitney.result(count_column="count", sample_column="borough")
        result_list.append(result)

    elif i["method"] == "kruskal-wallis-h-test":
        kruskal_wallis = KruskalWallisHTest(array=i["array"], q=i["q"])
        result = kruskal_wallis.result(count_column="count", sample_column="borough")
        result_list.append(result)

for i in result_list:
    print(i)