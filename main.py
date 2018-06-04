from etl import ETL
from kmeans_model import KMeansModel
import numpy as np
import pandas as pd

def main():
    etl = ETL()
    etl.extract()
    etl.transform()
    kmeans_model = KMeansModel(etl.observations,'modeling_text',
                               n_clusters=3,
                               n_features=1000)
    kmeans_model.vectorize()
    kmeans_model.apply_lsa(n_components=50)
    kmeans_model.run()
    kmeans_model.get_metrics()

# Main section
if __name__ == '__main__':
    main()