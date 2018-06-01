from etl import ETL
from kmeans_model import KMeansModel
import numpy as np
import pandas as pd

def main():
    etl = ETL()
    etl.extract()
    etl.transform()
    kmeans_model = KMeansModel(etl.observations,'modeling_text',
                               use_hashing=False,
                               use_idf=True,
                               n_clusters=3,
                               n_features=1000,
                               verbose=True)
    kmeans_model.vectorize()
    kmeans_model.apply_lsa(n_components=50)
    kmeans_model.run()
    kmeans_model.get_metrics()

    print(etl.observations[['category','cluster']].head(20))

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D



    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111, projection='3d')
    x = etl.observations['component_1']
    y = etl.observations['component_2']
    z = etl.observations['component_3']

    ax.scatter(x, y, z, marker="s", c=etl.observations['cluster'], s=40)

    for angle in range(0, 10):
        ax.view_init(30, 36*angle)
        #plt.draw()
        #plt.pause(.001)
        plt.show()

    df =  etl.observations[['component_1','component_2','component_3','category','cluster']].copy()
    df.to_csv('clustering.csv')

    etl.observations[['modeling_text_list']].to_csv('text.csv')

# Main section
if __name__ == '__main__':
    main()