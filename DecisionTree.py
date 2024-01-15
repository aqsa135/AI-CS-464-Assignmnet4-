
import pandas as pd
from collections import Counter
from math import log2 as log

def entropy(values):
    c = Counter(values)
    ent = 0
    distinct = set(values)

    for item in distinct:
        frequency = c[item] / len(values)
        ent += -1 * frequency * log(frequency)
    return ent


## takes as input two pandas series - one representing the variable
## to be tested, and the other the corresponding classifications.
## return the remainder - the weighted average of entropy.
## you do this. DONE
def remainder(variables, classifications):
    variables = pd.Series(variables)
    classifications = pd.Series(classifications)

    unique_vars = variables.unique()
    total_length = len(variables)

    weighted_entropy_sum = 0

    for var in unique_vars:
        subset_classification = classifications[variables == var]
        subset_entropy = entropy(subset_classification)

        weight = len(subset_classification) / total_length
        weighted_entropy_sum += weight * subset_entropy

    return weighted_entropy_sum

## df is a pandas dataframe, and classifications the corresponding
# classifications.
## check each column in the dataframe and return the column label
# of the column which maximizes gain (minimizes remainder.)    DONE

def select_attribute(df, classifications):
    min_remainder = float('inf')
    best_attrs = []

    for col in df.columns:
        rem = remainder(df[col], classifications)
        if rem < min_remainder:
            min_remainder = rem
            best_attrs = [col]
        elif rem == min_remainder:
            best_attrs.append(col)

    return random.choice(best_attrs)


class Node:
    def __init__(self, classification=None, attribute=None):
        self.classification = classification
        self.attribute = attribute
        self.children = {}

    def isLeaf(self):
        return len(self.children) == 0

## This is a recursive function.
## Base case #1. Our data has 0 entropy. We are done. Create and return
## a leaf node containing the value stored in the (right-hand) classification
## column.
## Base case #2. We are out of rows. There is no more data.
# Call ZeroR on the whole dataset and use this value.
## Base Case #3 We are out of columns. There is noise in our data.
# Call ZeroR on the whole dataset and use this value.
## Recursive step: Call select_attribute to find the attribute that maximizes
## gain (or minimizes remainder).
## Then, split your dataset. For each value of that attribute, select the rows
## that contain that attribute value, and construct a subtree (removing the selected attribute)
## That subtree is added to the children dictionary.
## Question: How do you deal with subtrees for missing values?
## When setting up your learning algorithm, create a dictionary that maps
## each attribute to all of its possible values. Then reference that
## to find all the possible values.   DONE
def make_tree(dataframe, classifications, attributes_values=None, depth=0, max_depth=5):
    if dataframe.empty or len(classifications) == 0:
        common_class = classifications.mode()[0] if len(classifications) > 0 else "default_value"
        return Node(classification=common_class)

    # Base case #1
    if entropy(classifications) == 0:
        return Node(classification=classifications.iloc[0])

    # Base case #2 and #3
    if dataframe.empty or dataframe.shape[1] == 0 or depth >= max_depth:
        common_class = classifications.mode()[0]
        return Node(classification=common_class)

    # Recursive step
    if attributes_values is None:
        attributes_values = {col: set(dataframe[col]) for col in dataframe.columns}

    split_attr = select_attribute(dataframe, classifications)
    root = Node(attribute=split_attr)

    for attr_val in attributes_values[split_attr]:
        subset = dataframe[dataframe[split_attr] == attr_val].reset_index(drop=True)
        subset_classifications = classifications[dataframe[split_attr] == attr_val].reset_index(drop=True)
        root.children[attr_val] = make_tree(subset.drop(columns=[split_attr]), subset_classifications,
                                            attributes_values, depth + 1, max_depth)
    return root


## Assume that we are being provided a pandas series named to_classify, and
## we are to return the classification for this data.
## This is also recursive.
## Base case. We are a leaf. Return tree.classification.
## Recursive step. What attribute do we test? Call classify on the child
# corresponding to the value of that attribute in tree.children DONE

def classify(tree, to_classify):
    if tree.isLeaf():
        return tree.classification

    attr_val = to_classify[tree.attribute]
    if attr_val in tree.children:
        return classify(tree.children[attr_val], to_classify)
    else:
        return tree.classification  # Default classification if attribute value is not found




from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import random


def five_fold_cross_validation(df, target_column_name):
    kf = KFold(n_splits=5, shuffle=True)
    accuracies = []

    for train_index, test_index in kf.split(df):
        train_data, test_data = df.iloc[train_index], df.iloc[test_index]

        tree = make_tree(train_data.drop(columns=[target_column_name]), train_data[target_column_name])

        true_values = test_data[target_column_name]
        predictions = [classify(tree, test_data.iloc[i].drop(labels=target_column_name)) for i in range(len(test_data))]

        # Debugging print statement
        # for pred, true_val in zip(predictions, true_values):
        #     print(pred, true_val)

        # Handling None
        predictions = ['default_value' if pred is None else pred for pred in predictions]  # Option 1 worked better

        # filtered_preds = [(pred, true_val) for pred, true_val in zip(predictions, true_values) if pred is not None]  # Option 2
        # predictions, true_values = zip(*filtered_preds)  # Option 2

        accuracies.append(accuracy_score(true_values, predictions))

    return sum(accuracies) / len(accuracies)


# Perform cross-validation on both datasets
tennis_df = pd.read_csv('tennis.csv')
restaurant_df = pd.read_csv('restaurant.csv')
restaurant_df.columns = restaurant_df.columns.str.strip()




tennis_accuracy = five_fold_cross_validation(tennis_df, 'play')
restaurant_accuracy = five_fold_cross_validation(restaurant_df, 'Will Wait')

print(f"Five-fold cross-validation accuracy for tennis dataset: {tennis_accuracy}")
print(f"Five-fold cross-validation accuracy for restaurant dataset: {restaurant_accuracy}")

# Load the breast cancer dataset
column_names = [
    "recurrence", "age", "menopause", "tumor-size", "inv-nodes", "node-caps",
    "deg-malig", "breast", "breast-quad", "irradiat"
]
breast_cancer_df = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data',
    names=column_names,
    header=None
)

# missing values if any.
breast_cancer_df = breast_cancer_df.replace('?', pd.NA)
breast_cancer_df = breast_cancer_df.dropna()

breast_cancer_accuracy = five_fold_cross_validation(breast_cancer_df, 'recurrence')
print(f"Five-fold cross-validation accuracy for breast cancer dataset: {breast_cancer_accuracy}")


from sklearn.tree import DecisionTreeClassifier


def sklearn_five_fold_cross_validation(df, target_column_name):
    kf = KFold(n_splits=5, shuffle=True)
    accuracies = []

    X = df.drop(columns=[target_column_name])
    y = df[target_column_name]

    for train_index, test_index in kf.split(df):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        clf = DecisionTreeClassifier(criterion='entropy')
        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)
        accuracies.append(accuracy_score(y_test, predictions))

    return sum(accuracies) / len(accuracies)



from sklearn.preprocessing import LabelEncoder

# copy of the dataframe
breast_cancer_df_encoded = breast_cancer_df.copy()

# Use label encoding on columns with string data
label_encoders = {}  # To store encoders for possible inverse transformations
for col in breast_cancer_df_encoded.columns:
    if breast_cancer_df_encoded[col].dtype == 'object':
        le = LabelEncoder()
        breast_cancer_df_encoded[col] = le.fit_transform(breast_cancer_df_encoded[col])
        label_encoders[col] = le


sklearn_accuracy = sklearn_five_fold_cross_validation(breast_cancer_df_encoded, 'recurrence')
print(f"Sklearn DecisionTreeClassifier accuracy for breast cancer dataset: {sklearn_accuracy}")

print(f"Decision tree accuracy: {breast_cancer_accuracy}")
print(f"Sklearn DecisionTreeClassifier accuracy: {sklearn_accuracy}")



from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def random_forest_cross_validation(df, target_column_name):
    X = df.drop(columns=[target_column_name])
    y = df[target_column_name]

    clf = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)

    accuracies = cross_val_score(clf, X, y, cv=5)

    return accuracies.mean()


custom_tree_accuracy = five_fold_cross_validation(breast_cancer_df_encoded, 'recurrence')
print(f"Custom DecisionTree accuracy for breast cancer dataset: {custom_tree_accuracy}")

# Sklearn DecisionTreeClassifier
sklearn_tree_accuracy = sklearn_five_fold_cross_validation(breast_cancer_df_encoded, 'recurrence')
print(f"Sklearn DecisionTreeClassifier accuracy for breast cancer dataset: {sklearn_tree_accuracy}")

# Sklearn RandomForestClassifier
random_forest_accuracy = random_forest_cross_validation(breast_cancer_df_encoded, 'recurrence')
print(f"Sklearn RandomForestClassifier accuracy for breast cancer dataset: {random_forest_accuracy}")



# References:
#https://towardsdatascience.com/decision-trees-in-machine-learning-641b9c4e8052
#https://victorzhou.com/blog/information-gain/
#https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/
#https://scikit-learn.org/stable/modules/cross_validation.html
#https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
#https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing


# if __name__ == "__main__":
#     df = pd.read_csv('tennis.csv')
#     data = df.drop(columns=['play'])
#     classifications = df['play']
#     tree = make_tree(data, classifications)
#
#     new_data = pd.Series({'outlook': 'sunny', 'temperature': 'hot', 'humidity': 'high', 'windy': 'FALSE'})
#     prediction = classify(tree, new_data)
#     print(f"Prediction for new data: {prediction}")