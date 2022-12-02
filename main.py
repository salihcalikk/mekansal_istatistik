from functions import ChiKare
import numpy as np

test_arrays = [{"method": "chi-kare", "q": 95, "array": np.array([[34, 63], [97, 16]])},
               {"method": "chi-kare", "q": 95, "array": np.array([[14, 43, 27, 45, 32], [16, 21, 34, 61, 15]])},
               {"method": "chi-kare", "q": 95, "array": np.array([[17, 2, 1, 1], [14, 3, 2, 1],
                                            [10, 1, 0, 0], [13, 2, 3, 2],
                                            [19, 3, 2, 0]])}]

result_list = []
for i in test_arrays:
    if i["method"] == "chi-kare":
        chi_kare = ChiKare(array=i["array"], q=i["q"])
        result = chi_kare.chi_kare_result()
        result_list.append(result)

for i in result_list:
    print(i)
