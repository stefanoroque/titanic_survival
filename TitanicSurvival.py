import tensorflow as tf
import pandas as pd
import re
import numpy as np
from sklearn import metrics


# Function that gets the title from each name and creates a new feature from it
def get_title(name):
    # Search for a word preceded by a space and followed by a period (this will be the title)
    title_search = re.search(" ([A-Za-z]+)\.", name)

    if title_search:
        return title_search.group(1)  # If title exists, return it
    else:
        return "-"  # else, just return a dash


# Function that creates pandas DataFrames from the .csv files
def read_in_csv():
    df = pd.read_csv("titanic_data.csv", sep=",")

    # Convert empty "age" slots to "-1"
    df["Age"].fillna(-1, inplace=True)

    # convert empty "cabin" and "embarked" slots to "-"
    df["Cabin"].fillna("-", inplace=True)
    df["Embarked"].fillna("-", inplace=True)

    # Get titles of each passenger in both data sets
    # Also replace uncommon titles and consolidate synonymous titles
    for passenger in df:
        df["Title"] = df["Name"].apply(get_title)
        df["Title"] = df["Title"].replace(
            ["Lady", "Countess", "Capt", "Col", "Don", "Dona", "Dr", "Major", "Rev", "Sir", "Jonkheer"], "Rare")
        df["Title"] = df["Title"].replace("Mlle", "Miss")
        df["Title"] = df["Title"].replace("Ms", "Miss")
        df["Title"] = df["Title"].replace("Mme", "Mrs")

    return df


# Prepare input features from the DataFrame
# Returns a DataFrame that contains the features to be used in the model
def preprocess_features(df):
    selected_features = df[
    ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked", "Title"]]
    processed_features = selected_features.copy()

    return processed_features


# Prepares target feature from the DataFrame
# Returns a DataFrame that contains the target feature (Whether the passenger survived or not)
def preprocess_target(df):
    output_targets = pd.DataFrame()  # This DataFrame must contain exactly 1 column
    output_targets["Survived"] = df["Survived"]

    return output_targets


# Construct TensorFlow Feature Columns
# Returns a set of feature columns
def construct_feature_columns(input_features):
    feature_columns = []

    age_numeric_fc = tf.feature_column.numeric_column("Age")
    sibsp_numeric_fc = tf.feature_column.numeric_column("SibSp")
    parch_numeric_fc = tf.feature_column.numeric_column("Parch")
    fare_numeric_fc = tf.feature_column.numeric_column("Fare")
    # Bucketize the numeric columns so they are compatible with the categorical columns
    age_fc = tf.feature_column.bucketized_column(source_column=age_numeric_fc,
                                                 boundaries=[0, 17, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75])
    sibsp_fc = tf.feature_column.bucketized_column(source_column=sibsp_numeric_fc,
                                                   boundaries=[1, 2, 3, 4, 5, 6])
    parch_fc = tf.feature_column.bucketized_column(source_column=parch_numeric_fc,
                                                   boundaries=[1, 2, 3, 4, 5, 6])
    fare_fc = tf.feature_column.bucketized_column(source_column=fare_numeric_fc,
                                                  boundaries=[8, 15, 30, 50])

    pclass_fc = tf.feature_column.categorical_column_with_identity(key="Pclass", num_buckets=4)
    sex_fc = tf.feature_column.categorical_column_with_vocabulary_list(key="Sex", vocabulary_list=["male", "female"])
    cabin_fc = tf.feature_column.categorical_column_with_hash_bucket(key="Cabin", hash_bucket_size=500)
    embarked_fc = tf.feature_column.categorical_column_with_vocabulary_list(key="Embarked", vocabulary_list=["C", "Q", "S"])
    title_fc = tf.feature_column.categorical_column_with_vocabulary_list(key="Title",
                                                                         vocabulary_list=["Master", "Miss", "Mrs", "Mr", "Rare"])

    feature_columns.append(age_fc)
    feature_columns.append(sibsp_fc)
    feature_columns.append(parch_fc)
    feature_columns.append(fare_fc)
    # Wrap categorical feature columns in indicator columns to give us a multi-hot representation of each feature
    feature_columns.append(tf.feature_column.indicator_column(pclass_fc))
    feature_columns.append(tf.feature_column.indicator_column(sex_fc))
    feature_columns.append(tf.feature_column.indicator_column(cabin_fc))
    feature_columns.append(tf.feature_column.indicator_column(embarked_fc))
    feature_columns.append(tf.feature_column.indicator_column(title_fc))

    return feature_columns


# Input function to help train the neural net
def input_funct(features, targets, batch_size=1, shuffle=False, num_epochs=None):

    # Convert pandas data into a dict of np arrays.
    features = {key: np.array(value) for key, value in dict(features).items()}

    # Construct a dataset, and configure batching/repeating.
    ds = tf.data.Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
        ds = ds.shuffle(10000)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

# Function that trains the neural net
# Returns a trained neural net
def train_model(
        learning_rate,
        steps,
        batch_size,
        feature_columns,
        training_features,
        training_target,
        validation_features,
        validation_target):

    periods = 10
    steps_per_period = steps / periods

    # Create a deep neural net classifier
    custom_optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate)
    custom_optimizer = tf.contrib.estimator.clip_gradients_by_norm(custom_optimizer, 5.0)
    nn_classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        optimizer=custom_optimizer,
        hidden_units=[30, 20],
        n_classes=2
    )

    training_input_funct = lambda: input_funct(training_features,
                                            training_target["Survived"],
                                            batch_size=batch_size)
    predict_training_input_funct = lambda: input_funct(training_features,
                                                    training_target["Survived"],
                                                    num_epochs=1,
                                                    shuffle=False)
    predict_validation_input_funct = lambda: input_funct(validation_features,
                                                      validation_target["Survived"],
                                                      num_epochs=1,
                                                      shuffle=False)

    # Train the model, but do so inside a loop so that we can periodically assess accuracy metrics
    print("Training model...")
    print("Accuracy (on training data):")
    training_acc_list = []
    validation_acc_list = []
    for period in range(1, periods+1):
        # Train the model, starting from the prior state
        nn_classifier.train(
            input_fn=training_input_funct,
            steps=steps_per_period
        )
        # Take a break and compute predictions
        training_predictions = nn_classifier.predict(input_fn=predict_training_input_funct)
        training_predictions = np.array([item["class_ids"][0] for item in training_predictions])

        validation_predictions = nn_classifier.predict(input_fn=predict_validation_input_funct)
        validation_predictions = np.array([item["class_ids"][0] for item in validation_predictions])

        # Compute training and validation accuracy
        training_accuracy = metrics.accuracy_score(training_predictions, training_target)
        validation_accuracy = metrics.accuracy_score(validation_predictions, validation_target)
        # Occasionally print the current accuracy
        print("  period %02d : %0.2f" % (period, training_accuracy))
        training_acc_list.append(training_accuracy)
        validation_acc_list.append(validation_accuracy)

    # Print accuracy on validation data aswell
    print("-----------------------------------------")
    print("Accuracy (on validation data):")
    for i in range(0, len(validation_acc_list)):
        print("  period %02d : %0.2f" % (i+1, validation_acc_list[i]))
    print("-----------------------------------------")
    print("Model training finished.")

    return nn_classifier

def main():

    # Read in the data
    titanic_df = read_in_csv()

    # Print DataFrame so we can see what the model is being fed
    print(titanic_df.to_string())
    print("-----------------------------------------")

    # Choose the first 640 passengers for training
    training_features = preprocess_features(titanic_df.head(640))
    training_target = preprocess_target(titanic_df.head(640))

    # Choose the last 250 passengers for validation
    validation_features = preprocess_features(titanic_df.tail(250))
    validation_target = preprocess_target(titanic_df.tail(250))

    # Train the model and return it
    trained_model = train_model(learning_rate=0.25, steps=200, batch_size=100,
                                feature_columns=construct_feature_columns(training_features),
                                training_features=training_features,
                                training_target=training_target,
                                validation_features=validation_features,
                                validation_target=validation_target)


main()
