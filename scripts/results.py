import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 2000)
pd.set_option("display.max_colwidth", None)

if __name__ == "__main__":
    df = pd.read_csv("./rouge_data/results.csv", index_col=0)
    print(df.describe(percentiles=[]))
