import pandas as pd
import os

lsi = pd.read_csv(os.path.join(os.getcwd(), 'spark_lsi', "spark_lsi_index.csv"), index_col=['review_id'])
# spark_lsi = pd.read_csv(os.path.join(os.getcwd(), 'spark_lsi', "spark_lsi.csv"))
# spark_lsi['review_id'] = lsi.index.values
# spark_lsi.set_index(['review_id'], inplace=True)
# spark_lsi.to_csv(os.path.join(os.getcwd(), 'spark_lsi', "spark_lsi_index.csv"))