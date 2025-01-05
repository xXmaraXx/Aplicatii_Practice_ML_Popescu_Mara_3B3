import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from graphviz import Digraph

def visualize_tree(tree, tree_name="Decision Tree"):
    """
    Creează o reprezentare grafică a arborelui decizional folosind Graphviz.

    Parameters:
        tree (dict): Structura arborelui decizional.
        tree_name (str): Numele arborelui pentru vizualizare.

    Returns:
        Digraph: Obiect Graphviz care reprezintă arborele decizional.
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

def calculate_entropy_reduction(data, target, feature):
    """
    Calculează reducerea entropiei pentru un feature dat.

    Parameters:
        data (DataFrame): Datele de intrare.
        target (str): Numele coloanei țintă.
        feature (str): Numele caracteristicii analizate.

    Returns:
        tuple: Reducerea entropiei și valoarea mediană a caracteristicii.
    """
    def entropy(values):
        probabilities = values / values.sum()
        probabilities = probabilities[probabilities > 0]
        return -np.sum(probabilities * np.log2(probabilities))

    total_entropy = entropy(np.histogram(data[target], bins=20)[0]) * len(data)

    median_value = data[feature].median()
    left_split = data[data[feature] <= median_value]
    right_split = data[data[feature] > median_value]

    left_entropy = entropy(np.histogram(left_split[target], bins=20)[0]) * len(left_split) if len(left_split) > 0 else 0
    right_entropy = entropy(np.histogram(right_split[target], bins=20)[0]) * len(right_split) if len(right_split) > 0 else 0

    entropy_reduction = total_entropy - (left_entropy + right_entropy)
    return entropy_reduction, median_value

def build_tree_with_entropy(data, target, features, max_depth, min_samples_split, depth=0):
    """
    Construiește un arbore decizional folosind reducerea entropiei ca metrică.

    Parameters:
        data (DataFrame): Datele de antrenament.
        target (str): Numele coloanei țintă.
        features (list): Lista caracteristicilor candidate pentru split-uri.
        max_depth (int): Adâncimea maximă a arborelui.
        min_samples_split (int): Numărul minim de mostre pentru a realiza split-ul.
        depth (int): Adâncimea curentă (default: 0).

    Returns:
        dict: Structura arborelui decizional.
    """
    if depth >= max_depth or len(data) < min_samples_split:
        return {"leaf": np.mean(data[target])}

    best_feature, best_value, best_reduction = None, None, -float("inf")
    for feature in features:
        reduction, value = calculate_entropy_reduction(data, target, feature)
        if reduction > best_reduction:
            best_feature, best_value, best_reduction = feature, value, reduction

    if best_feature is None:
        return {"leaf": np.mean(data[target])}

    left_split = data[data[best_feature] <= best_value]
    right_split = data[data[best_feature] > best_value]

    return {
        "feature": best_feature,
        "value": best_value,
        "left": build_tree_with_entropy(left_split, target, features, max_depth, min_samples_split, depth + 1),
        "right": build_tree_with_entropy(right_split, target, features, max_depth, min_samples_split, depth + 1),
    }

def predict_tree(row, tree):
    """
    Realizează o predicție pentru un rând de date pe baza unui arbore decizional.

    Parameters:
        row (Series): Un rând de date.
        tree (dict): Structura arborelui decizional.

    Returns:
        float: Predicția calculată.
    """
    if "leaf" in tree:
        return tree["leaf"]
    if row[tree["feature"]] <= tree["value"]:
        return predict_tree(row, tree["left"])
    else:
        return predict_tree(row, tree["right"])

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
    print(f"Training model for {feature}...")
    tree = build_tree_with_entropy(train_data, feature, features, max_depth=5, min_samples_split=10)
    predictions[feature] = test_data.apply(lambda row: predict_tree(row, tree), axis=1)
    tree_graph = visualize_tree(tree, tree_name=f"{feature} arbore")
    tree_graph.render(f"decision_tree_{feature}", format="png", cleanup=True)
    tree_graph.view()

test_data['Pred_Productie[MW]'] = predictions['Productie[MW]']
test_data['Pred_Consum[MW]'] = predictions['Consum[MW]']
test_data['Pred_Sold[MW]'] = test_data['Pred_Consum[MW]'] - test_data['Pred_Productie[MW]']

test_data['Pred_Sold[MW]'] = test_data['Pred_Sold[MW]'].astype(int)

mse_sold = mean_squared_error(test_data['Sold[MW]'], test_data['Pred_Sold[MW]'])
print(f"MSE final pentru Sold[MW]: {mse_sold}")

output_path = "test_predictions_entropy_decembrie_2024.xlsx"
test_data.to_excel(output_path, index=False)

real_total_sold = test_data['Sold[MW]'].sum()
predicted_total_sold = test_data['Pred_Sold[MW]'].sum()
print(f"Sold real pentru decembrie 2024: {real_total_sold} MW")
print(f"Sold prezis pentru decembrie 2024: {predicted_total_sold} MW")
print(f"Difference: {predicted_total_sold - real_total_sold} MW")

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
