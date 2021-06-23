from sklearn import tree
import pydotplus
import base64
from IPython.display import Image
import os
from pathlib import Path
import pandas as pd


class CustomUtils:
    @staticmethod
    def get_base64_encoded_image(decision_tree, columns):
        dot_data = tree.export_graphviz(decision_tree, out_file=None, feature_names=columns, impurity=False,
                                        filled=True,
                                        proportion=True,
                                        rounded=True)

        graph = pydotplus.graph_from_dot_data(dot_data)
        image = Image(graph.create_png())
        encodedImage = base64.b64encode(image.data).decode("utf-8")
        return encodedImage

    @staticmethod
    def read_file_and_return_df(filename):
        filepath = os.path.join(Path(__file__).parent, 'data', '.')
        df = pd.read_csv(f'{filepath}/{filename}')
        return df

