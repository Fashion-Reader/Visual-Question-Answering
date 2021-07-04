# ensemble.py
import pandas as pd
from glob import glob
from scipy.stats import mode

def main():
    # main submission
    result_dataframe = pd.read_csv("./results/train/train_v1/submission_v1.csv")["answer"]

    # ensemble할 csv files
    submission_dirs = glob("./results/train/*/*_v[23].csv")
    
    print(submission_dirs)

    # 전체 csv 갯수
    N = len(submission_dirs) + 1

    # 하나의 dataframe으로 concat
    for submission_dir in submission_dirs:
        result_dataframe = pd.concat(
            [result_dataframe, pd.read_csv(submission_dir)["answer"]], axis=1
        )

    # 사용한 csv file 갯수 확인
    print("Ensemble에 사용된 submission 갯수: ", result_dataframe.shape[1])

    # 최빈값 찾아주기
    # 모두 다른 경우 main submission 값 선택
    # 주의: [1,1,2,2] 인 경우 무조건 작은 숫자를 선택
    # result 결과 3개 or 5개 사용 권장
    def find_mode(data):
        if len(set(data)) == N:
            return data[0]

        return mode(data)[0][0]

    result = pd.DataFrame([])
    result["answer"] = result_dataframe.apply(find_mode, axis=1)
    result = result.reset_index()
    result.columns = ["ID", "answer"]
    
    result.to_csv(
        "./results/train/submission_ensemble.csv", index=False
    )


if __name__ == "__main__":
    main()