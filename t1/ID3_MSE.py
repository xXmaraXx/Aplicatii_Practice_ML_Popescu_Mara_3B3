import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from graphviz import Digraph

def visualize_tree(tree, tree_name="Decision Tree"):
    """
    Creeaza o reprezentare grafica a arborelui decizional.
    """
    dot = Digraph(comment=tree_name)
    node_id = [0]

    def add_nodes_edges(node, parent=None, edge_label=None):
        current_id = node_id[0]
        node_id[0] += 1

        if "leaf" in node:
            dot.node(str(current_id), f"Leaf\n{node['leaf']:.2f}", shape="box", style="filled", color="lightblue")
        else:
            feature = node["feature"]
            value = node["value"]
            dot.node(str(current_id), f"{feature} <= {value:.2f}", shape="ellipse", style="filled", color="lightyellow")

        if parent is not None:
            dot.edge(str(parent), str(current_id), label=edge_label if edge_label else "")

        if "left" in node:
            add_nodes_edges(node["left"], current_id, "True")
        if "right" in node:
            add_nodes_edges(node["right"], current_id, "False")

    add_nodes_edges(tree)
    return dot




def calculate_mse_reduction(data, target, feature):
    """
    Calculează reducerea erorii medii patrate (MSE) pentru o caracteristică dată.

    :param data: DataFrame-ul care conține datele de intrare.
    :param target: Numele coloanei care reprezintă variabila țintă.
    :param feature: Numele caracteristicii pentru care se calculează reducerea MSE.
    :return: Reducerea MSE și valoarea mediană utilizată pentru split.
    """
    total_mse = np.var(data[target]) * len(data)

    median_value = data[feature].median()
    left_split = data[data[feature] <= median_value]
    right_split = data[data[feature] > median_value]

    left_mse = np.var(left_split[target]) * len(left_split) if len(left_split) > 0 else 0
    right_mse = np.var(right_split[target]) * len(right_split) if len(right_split) > 0 else 0

    mse_reduction = total_mse - (left_mse + right_mse)
    return mse_reduction, median_value


class ID3_MSE:
    def __init__(self, max_depth=5, min_samples_split=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, data, target, features):
        """
        Construiește arborele de decizie pe baza datelor de antrenament.

        :param data: DataFrame-ul de antrenament.
        :param target: Numele coloanei țintă.
        :param features: Lista de caracteristici utilizate pentru split.
        :return: Structura arborelui.
        """
        self.tree = self._build_tree(data, target, features, depth=0)
        return self.tree

    def _build_tree(self, data, target, features, depth):
        """
        Recursiv construiește arborele de decizie.

        :param data: Datele curente pentru acest nod.
        :param target: Numele coloanei țintă.
        :param features: Lista de caracteristici disponibile pentru split.
        :param depth: Adâncimea curentă a arborelui.
        :return: Structura nodului curent.
        """
        if depth >= self.max_depth or len(data) < self.min_samples_split:
            return {"leaf": np.mean(data[target])}

        best_feature, best_value, best_reduction = None, None, -float("inf")
        for feature in features:
            reduction, value = calculate_mse_reduction(data, target, feature)
            if reduction > best_reduction:
                best_feature, best_value, best_reduction = feature, value, reduction

        if best_feature is None:
            return {"leaf": np.mean(data[target])}

        left_split = data[data[best_feature] <= best_value]
        right_split = data[data[best_feature] > best_value]

        return {
            "feature": best_feature,
            "value": best_value,
            "left": self._build_tree(left_split, target, features, depth + 1),
            "right": self._build_tree(right_split, target, features, depth + 1),
        }

    def predict(self, data):
        """
        Realizează predicții pentru fiecare instanță din setul de date.

        :param data: DataFrame-ul pentru care se fac predicții.
        :return: Un array cu predicțiile.
        """
        predictions = [self._predict_row(row, self.tree) for _, row in data.iterrows()]
        return np.array(predictions)

    def _predict_row(self, row, tree):
        """
        Predicție pentru o singură instanță.

        :param row: Instanța curentă (o linie din DataFrame).
        :param tree: Structura arborelui curent.
        :return: Valoarea prezisă.
        """
        if "leaf" in tree:
            return tree["leaf"]
        if row[tree["feature"]] <= tree["value"]:
            return self._predict_row(row, tree["left"])
        else:
            return self._predict_row(row, tree["right"])


file_path = "Grafic_SEN.xlsx"
data = pd.read_excel(file_path)

features = ['Consum[MW]', 'Productie[MW]', 'Carbune[MW]', 'Hidrocarburi[MW]',
            'Ape[MW]', 'Nuclear[MW]', 'Eolian[MW]', 'Foto[MW]', 'Biomasa[MW]']
target = 'Sold[MW]'
data[features + [target]] = data[features + [target]].apply(pd.to_numeric, errors='coerce')
data = data.dropna()

train_data = data[~data['Data'].str.contains("-12-")]
test_data = data[data['Data'].str.contains("-12-2024")]

models = {}
predictions = {}
for feature in features:
    print(f"Antrenăm modelul pentru {feature}...")
    dt_regressor = ID3_MSE(max_depth=5, min_samples_split=10)
    dt_regressor.fit(train_data, feature, features)
    predictions[feature] = dt_regressor.predict(test_data)
    tree_graph = visualize_tree(dt_regressor.tree, tree_name=f"{feature} arbore")
    tree_graph.render(f"dt_{feature}", format="png", cleanup=True)
    tree_graph.view()

test_data['Pred_Productie[MW]'] = predictions['Productie[MW]']
test_data['Pred_Consum[MW]'] = predictions['Consum[MW]']
test_data['Pred_Sold[MW]'] = test_data['Pred_Consum[MW]'] - test_data['Pred_Productie[MW]']

test_data['Pred_Sold[MW]'] = test_data['Pred_Sold[MW]'].astype(int)

mse_sold = mean_squared_error(test_data['Sold[MW]'], test_data['Pred_Sold[MW]'])
print(f"MSE final pentru Sold[MW]: {mse_sold}")

output_path = "test_predictions_decembrie_2024.xlsx"
test_data.to_excel(output_path, index=False)
real_total_sold = test_data['Sold[MW]'].sum()
predicted_total_sold = test_data['Pred_Sold[MW]'].sum()

print(f"Soldul total real pentru decembrie 2024: {real_total_sold} MW")
print(f"Soldul total prezis pentru decembrie 2024: {predicted_total_sold} MW")
print(f"Diferenta de: {predicted_total_sold - real_total_sold} MW")



import matplotlib.pyplot as plt

test_data_28_days = test_data.iloc[:28]
real_sold_28 = test_data_28_days['Sold[MW]'].tolist()
predicted_sold_28 = test_data_28_days['Pred_Sold[MW]'].tolist()

days_28 = list(range(1, 29))

plt.figure(figsize=(12, 6))
plt.plot(days_28, real_sold_28, label="Sold Real [MW]", marker="o", color="blue")
plt.plot(days_28, predicted_sold_28, label="Sold Prezis [MW]", marker="s", linestyle="--", color="orange")
plt.title("Comparație între Sold-ul Real și cel Prezis - Primele 28 de zile din Decembrie 2024")
plt.xlabel("Ziua lunii")
plt.ylabel("Sold [MW]")
plt.xticks(days_28)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
