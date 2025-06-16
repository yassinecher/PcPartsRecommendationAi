import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import os
import logging
import csv
from collections import defaultdict
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# List of CSV files
CSV_FILES = [
    "case_components.csv", "case_fan_components.csv", "cpu_components.csv",
    "cpu_cooler_components.csv", "custom_components.csv", "external_storage_components.csv",
    "fan_controller_components.csv", "headphones_components.csv", "keyboard_components.csv",
    "memory_components.csv", "monitor_components.csv", "motherboard_components.csv",
    "mouse_components.csv", "operating_system_components.csv", "optical_drive_components.csv",
    "power_supply_components.csv", "sound_card_components.csv",
    "speakers_components.csv", "storage_components.csv", "thermal_components.csv",
    "ups_components.csv", "video_card_components.csv", "webcam_components.csv",
    "wired_network_adapter_components.csv", "wireless_network_adapter_components.csv"
]

# Define critical compatibility fields
COMPATIBILITY_FIELDS = {
    "CPU": ["socket", "core_count"],
    "Motherboard": ["cpu_socket", "memory_type", "form_factor"],
    "Memory": ["memory_type", "speed"],
    "Power Supply": ["wattage", "efficiency_rating"],
    "Video Card": ["tdp", "length"],
    "Storage": ["interface", "form_factor"],
}

# Numeric fields for distance-based similarity
NUMERIC_FIELDS = ["core_count", "speed", "wattage", "tdp", "length"]


# Function to load and preprocess data
def load_data():
    all_data = []
    skipped_lines = defaultdict(int)
    for csv_file in CSV_FILES:
        if not os.path.exists(csv_file):
            logging.warning(f"{csv_file} does not exist, skipping...")
            continue
        try:
            df = pd.read_csv(csv_file, quoting=csv.QUOTE_ALL, on_bad_lines='skip', encoding="utf-8-sig")
            if len(df) == 0:
                logging.warning(f"No valid data in {csv_file} after skipping bad lines, skipping...")
                continue
            if "component_type" not in df.columns or "component_name" not in df.columns:
                logging.warning(f"Missing required columns in {csv_file}, skipping...")
                continue
            # Standardize column names
            df.columns = [col.lower().replace(" ", "_") for col in df.columns]
            # Fill missing component_type and standardize
            if df['component_type'].isna().any():
                inferred_type = csv_file.replace("_components.csv", "").replace("_", " ").title()
                df['component_type'] = df['component_type'].fillna(inferred_type)
            df['component_type'] = df['component_type'].str.title()
            # Log component_type values for debugging
            logging.info(f"Component types in {csv_file}: {df['component_type'].unique()}")
            # Normalize compatibility fields
            for field in ['socket', 'cpu_socket', 'memory_type', 'interface', 'form_factor', 'efficiency_rating']:
                if field in df.columns:
                    df[field] = df[field].str.strip().str.replace(" ", "").str.upper()
                else:
                    logging.warning(f"Field '{field}' not found in {csv_file}")
            # Convert numeric fields to float
            for field in NUMERIC_FIELDS:
                if field in df.columns:
                    df[field] = pd.to_numeric(df[field], errors='coerce')
                else:
                    logging.warning(f"Field '{field}' not found in {csv_file}")
            # Log unique values for key fields
            if 'socket' in df.columns and 'CPU' in df['component_type'].values:
                logging.info(
                    f"Unique socket values in {csv_file}: {df[df['component_type'] == 'CPU']['socket'].unique()}")
            if 'cpu_socket' in df.columns and 'Motherboard' in df['component_type'].values:
                logging.info(
                    f"Unique cpu_socket values in {csv_file}: {df[df['component_type'] == 'Motherboard']['cpu_socket'].unique()}")
            all_data.append(df)
            skipped_lines[csv_file] += sum(1 for _ in open(csv_file, 'r', encoding="utf-8-sig")) - len(df) - 1
        except Exception as e:
            logging.error(f"Error loading {csv_file}: {e}")
            continue

    if not all_data:
        raise ValueError("No valid data loaded from CSV files.")

    combined_df = pd.concat(all_data, ignore_index=True)
    for file, lines in skipped_lines.items():
        if lines > 0:
            logging.info(f"Skipped {lines} lines in {file} due to parsing issues.")
    logging.info(f"Loaded {len(combined_df)} components from CSV files.")
    return combined_df


# Function to build a user-item matrix for collaborative filtering
def build_user_item_matrix(df):
    df = df.assign(present=1)
    user_item_matrix = pd.pivot_table(
        df,
        values='present',
        index=df.index,
        columns='component_name',
        aggfunc='count',
        fill_value=0
    )
    return user_item_matrix


# Function to perform collaborative filtering using SVD
def collaborative_filtering(user_item_matrix, n_components=10):
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    matrix_reduced = svd.fit_transform(user_item_matrix)
    component_scores = svd.components_
    predicted_scores = np.dot(matrix_reduced, component_scores)
    if predicted_scores.max() != predicted_scores.min():
        predicted_scores = (predicted_scores - predicted_scores.min()) / (
                    predicted_scores.max() - predicted_scores.min())
    else:
        predicted_scores = np.ones_like(predicted_scores) * 0.5
    return matrix_reduced, component_scores, user_item_matrix


# Function to compute content-based similarity
def compute_content_similarity(df):
    # Initialize lists for features
    categorical_features = []
    numeric_features = []
    available_fields = {}

    # Process each component type
    for component_type, fields in COMPATIBILITY_FIELDS.items():
        subset = df[df['component_type'] == component_type].copy()
        available = []
        cat_subset_features = []
        num_subset_features = []
        for field in fields:
            if field in subset.columns:
                available.append(field)
                if field in NUMERIC_FIELDS:
                    subset[field] = pd.to_numeric(subset[field], errors='coerce').fillna(0)
                    num_subset_features.append(subset[[field]])
                else:
                    subset[field] = subset[field].fillna('Unknown').astype(str)
                    cat_subset_features.append(pd.get_dummies(subset[field], prefix=f"{component_type}_{field}"))
            else:
                logging.warning(f"Field '{field}' not found for {component_type} in dataset.")
        available_fields[component_type] = available

        # Combine features for this subset
        if cat_subset_features:
            categorical_features.append(pd.concat(cat_subset_features, axis=1).reindex(df.index, fill_value=0))
        if num_subset_features:
            numeric_features.append(pd.concat(num_subset_features, axis=1).reindex(df.index, fill_value=0))

    if not (categorical_features or numeric_features):
        logging.warning("No compatibility features found for content-based filtering.")
        return None, None

    # Compute categorical similarity
    if categorical_features:
        cat_feature_matrix = pd.concat(categorical_features, axis=1).fillna(0)
        cat_similarity = cosine_similarity(cat_feature_matrix)
        if cat_similarity.max() != cat_similarity.min():
            cat_similarity = (cat_similarity - cat_similarity.min()) / (cat_similarity.max() - cat_similarity.min())
        else:
            cat_similarity = np.ones_like(cat_similarity) * 0.1
    else:
        cat_similarity = np.ones((len(df), len(df))) * 0.1

    # Compute numeric similarity
    if numeric_features:
        num_feature_matrix = pd.concat(numeric_features, axis=1).fillna(0)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(num_feature_matrix)
        distances = np.linalg.norm(scaled_features[:, None] - scaled_features[None, :], axis=2)
        num_similarity = 1 / (1 + distances)
        if num_similarity.max() != num_similarity.min():
            num_similarity = (num_similarity - num_similarity.min()) / (num_similarity.max() - num_similarity.min())
        else:
            num_similarity = np.ones_like(num_similarity) * 0.1
    else:
        num_similarity = np.ones((len(df), len(df))) * 0.1

    # Combine similarities
    similarity_matrix = 0.7 * cat_similarity + 0.3 * num_similarity
    if similarity_matrix.max() != similarity_matrix.min():
        similarity_matrix = (similarity_matrix - similarity_matrix.min()) / (
                    similarity_matrix.max() - similarity_matrix.min())
    else:
        similarity_matrix = np.ones_like(similarity_matrix) * 0.1

    logging.info(f"Available compatibility fields: {available_fields}")
    return similarity_matrix, df.index


# Function to recommend components
def recommend_components(input_components, df, user_item_matrix, matrix_reduced, component_scores,
                         similarity_matrix=None, similarity_index=None):
    input_indices = []
    for comp in input_components:
        if comp in user_item_matrix.columns:
            input_indices.append(user_item_matrix.columns.get_loc(comp))
        else:
            logging.warning(f"Component {comp} not found in the dataset.")
            return "No matching components found in the dataset."

    input_types = set(df[df['component_name'].isin(input_components)]['component_type'].str.lower())

    build_vector = np.zeros(user_item_matrix.shape[0])
    build_vector[0] = 1
    predicted_scores = np.dot(matrix_reduced[0, :], component_scores)

    # Exclude input components
    for idx in input_indices:
        predicted_scores[idx] = -np.inf

    # Adjust scores with content-based similarity
    if similarity_matrix is not None and similarity_index is not None:
        for comp in input_components:
            if comp in df['component_name'].values:
                comp_idx = df[df['component_name'] == comp].index[0]
                if comp_idx in similarity_index:
                    sim_idx = list(similarity_index).index(comp_idx)
                    sim_scores = similarity_matrix[sim_idx]
                    for i, score in enumerate(sim_scores):
                        orig_idx = similarity_index[i]
                        comp_name = df.loc[orig_idx, 'component_name']
                        if comp_name in user_item_matrix.columns:
                            matrix_idx = user_item_matrix.columns.get_loc(comp_name)
                            predicted_scores[matrix_idx] += score * 2.0
                    logging.info(f"Similarity scores for {comp}: {sim_scores[:5]}...")

    # Normalize final scores to 0-10 range
    if predicted_scores.max() != predicted_scores.min():
        predicted_scores = 10 * (predicted_scores - predicted_scores.min()) / (
                    predicted_scores.max() - predicted_scores.min())
    else:
        predicted_scores = np.ones_like(predicted_scores) * 5.0

    # Select diverse recommendations
    recommendations = []
    seen_types = set()
    all_indices = np.argsort(predicted_scores)[::-1]
    for idx in all_indices:
        comp_name = user_item_matrix.columns[idx]
        if comp_name in input_components:
            continue
        comp_type = df[df['component_name'] == comp_name]['component_type'].iloc[0].lower()
        if comp_type in input_types:
            continue
        if comp_type not in seen_types:
            recommendations.append((comp_type.title(), comp_name, predicted_scores[idx]))
            seen_types.add(comp_type)
        if len(recommendations) >= 5:
            break

    if not recommendations:
        return "No compatible components found after excluding input component type."

    logging.info(
        f"Final predicted scores: {[(n, s) for n, s in zip(user_item_matrix.columns, predicted_scores) if s > 0][:5]}...")
    return recommendations


# Main function
def main():
    try:
        df = load_data()
        user_item_matrix = build_user_item_matrix(df)
        matrix_reduced, component_scores, user_item_matrix = collaborative_filtering(user_item_matrix)
        similarity_matrix, similarity_index = compute_content_similarity(df)

        input_components = ["Intel Core i9-13900K 3 GHz 24-Core"]
        logging.info(f"Input components: {input_components}")
        recommendations = recommend_components(
            input_components,
            df,
            user_item_matrix,
            matrix_reduced,
            component_scores,
            similarity_matrix,
            similarity_index
        )

        if isinstance(recommendations, str):
            print(recommendations)
        else:
            print("Recommended components:")
            for comp_type, comp_name, score in recommendations:
                print(f"- {comp_type}: {comp_name} (Score: {score:.2f})")
    except Exception as e:
        logging.error(f"Error in main: {e}")
        raise e


if __name__ == "__main__":
    main()