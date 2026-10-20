from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from model_utils import (
    CATEGORICAL_FEATURES,
    FEATURE_COLS,
    NUMERIC_FEATURES,
    REQUIRED_COLUMNS,
    TARGET_COL,
    load_model_artifact,
    preprocess_data,
    validate_columns,
)


st.set_page_config(
    page_title="Restaurant Rating Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

sns.set_style("whitegrid")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Dataset.csv"
ARTIFACT_PATH = BASE_DIR / "model_artifacts" / "restaurant_model.joblib"

COUNTRY_NAMES = {
    1: "India", 14: "Australia", 30: "Brazil", 37: "Canada",
    94: "Indonesia", 148: "New Zealand", 162: "Philippines",
    166: "Qatar", 184: "Singapore", 189: "South Africa",
    191: "Sri Lanka", 208: "Turkey", 214: "UAE",
    215: "United Kingdom", 216: "United States",
}


st.markdown(
    """
    <style>
        html, body, [class*="css"] {
            font-size: 18px;
        }
        .main-title {
            font-size: 2.6rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            font-size: 1.2rem;
            color: #888;
            margin-bottom: 1.5rem;
        }
        .prediction-card {
            padding: 1.2rem;
            border-radius: 12px;
            border: 1px solid #ddd;
            background: rgba(128, 128, 128, 0.08);
            text-align: center;
        }
        .prediction-value {
            font-size: 2.5rem;
            font-weight: 700;
        }

        /* Section headers */
        h1 { font-size: 2.4rem !important; }
        h2 { font-size: 1.9rem !important; }
        h3 { font-size: 1.5rem !important; }

        /* Widget labels (City, Country Code, Cuisines, etc.) */
        label, .stMarkdown p, .stCaption, [data-testid="stWidgetLabel"] p {
            font-size: 1.15rem !important;
        }

        /* Text inside dropdowns, inputs, radio buttons */
        .stSelectbox div, .stMultiSelect div, .stNumberInput input,
        .stRadio label, .stTextInput input {
            font-size: 1.1rem !important;
        }

        /* Buttons */
        .stButton button {
            font-size: 1.2rem !important;
            padding: 0.6rem 1rem;
        }

        /* Metrics (Predicted Rating, Model, etc.) */
        [data-testid="stMetricLabel"] {
            font-size: 1.1rem !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 2.1rem !important;
        }

        /* Static tables (st.table) get larger, more readable cells */
        table {
            font-size: 1.1rem !important;
        }
        table th, table td {
            padding: 0.6rem 0.9rem !important;
        }

        /* Info / success / warning / error banners */
        .stAlert p {
            font-size: 1.1rem !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    validate_columns(data, REQUIRED_COLUMNS)
    return data


@st.cache_data
def get_unique_cuisines(df, country_code=None):
    subset = df if country_code is None else df[df["Country Code"] == country_code]
    exploded = subset["Cuisines"].dropna().str.split(", ").explode().str.strip()
    return sorted(exploded[exploded != ""].unique().tolist())


st.markdown(
    '<div class="main-title">Restaurant Rating Predictor</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">'
    "Predict restaurant ratings with machine learning and explore the data"
    "</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Choose a section",
        [
            "Prediction",
            "Model Comparison",
            "Dataset Overview",
            "Exploratory Analysis",
        ],
    )


try:
    raw_df = load_data()
    df = preprocess_data(raw_df)
    artifact = load_model_artifact(ARTIFACT_PATH)
    models = artifact["models"]
    cv_results = artifact["cv_results"]
    test_results = artifact["test_results"]
    best_model_name = artifact["best_model_name"]
    best_model = models[best_model_name]
    city_coordinates = artifact["city_coordinates"]
except FileNotFoundError as exc:
    if Path(exc.filename or "").resolve() == DATA_PATH.resolve():
        st.error("Dataset.csv was not found. Place it beside app.py.")
    else:
        st.error(
            "The trained model artifact was not found. Run the final training/export "
            "cell in restaurant_predict.ipynb to create "
            "model_artifacts/restaurant_model.joblib, then restart the app."
        )
    st.stop()
except ValueError as exc:
    st.error(str(exc))
    st.stop()


if page == "Prediction":
    st.header("Predict Restaurant Rating")

    st.info(
        f"Best model by cross-validation (lowest mean MAE): **{best_model_name}**."
    )

    col1, col2 = st.columns(2)

    with col1:
        average_cost = st.number_input(
            "Average Cost for two (local currency)",
            min_value=1.0,
            value=float(df["Average Cost for two"].median()),
            step=10.0,
            help="Enter the cost in the currency used in the selected country.",
        )

        price_range = st.selectbox(
            "Price range",
            sorted(df["Price range"].dropna().unique().tolist()),
            index=1 if df["Price range"].nunique() > 1 else 0,
        )

        votes = st.number_input(
            "Votes",
            min_value=0,
            value=int(df["Votes"].median()),
            step=1,
        )

        table_booking = st.selectbox(
            "Has Table booking",
            ["Yes", "No"],
        )

    with col2:
        online_delivery = st.selectbox(
            "Has Online delivery",
            ["Yes", "No"],
        )

        country_options = sorted(
            df["Country Code"].dropna().unique().tolist()
        )
        country_code = st.selectbox(
            "Country Code",
            country_options,
            format_func=lambda c: f"{COUNTRY_NAMES.get(c, 'Unknown')} ({c})",
        )

        cuisine_options = get_unique_cuisines(df, country_code=country_code)
        default_cuisine = (
            ["North Indian"]
            if "North Indian" in cuisine_options
            else cuisine_options[:1]
        )
        selected_cuisines = st.multiselect(
            "Cuisines",
            cuisine_options,
            default=default_cuisine,
            help="Only cuisines served in the selected country are shown.",
        )

        city_options = sorted(
            df.loc[df["Country Code"] == country_code, "City"].dropna().unique().tolist()
        )
        city = st.selectbox(
            "City",
            city_options,
            help="The model also uses the location, taken as the median coordinates of this city.",
        )

    model_name = st.selectbox(
        "Prediction model",
        list(models.keys()),
        index=list(models.keys()).index(best_model_name),
    )

    number_of_cuisines = len(selected_cuisines)
    coordinate_key = (country_code, city)
    try:
        latitude = float(city_coordinates.loc[coordinate_key, "Latitude"])
        longitude = float(city_coordinates.loc[coordinate_key, "Longitude"])
    except KeyError:
        st.error("Coordinates are unavailable for this country and city. Retrain the model artifact.")
        st.stop()

    # One row with exactly the columns and types the pipelines were trained on.
    input_data = pd.DataFrame(
        [
            {
                "Average Cost for two": average_cost,
                "Price range": price_range,
                "Votes": votes,
                "Latitude": latitude,
                "Longitude": longitude,
                "Number of Cuisines": number_of_cuisines,
                "City": city,
                "Country Code": country_code,
                "Has Table booking": table_booking,
                "Has Online delivery": online_delivery,
            }
        ]
    )[FEATURE_COLS]

    if st.button("Predict Rating", type="primary", width="stretch"):
        if not selected_cuisines:
            st.warning("Please select at least one cuisine before predicting.")
        else:
            selected_model = models[model_name]
            prediction = float(selected_model.predict(input_data)[0])

            # Ratings in the source data are on a 0-5 scale.
            prediction = max(0.0, min(5.0, prediction))

            st.markdown("---")
            st.subheader("Prediction Result")

            result_col1, result_col2, result_col3 = st.columns(3)

            with result_col1:
                st.metric("Predicted Rating", f"{prediction:.2f} / 5.00")

            with result_col2:
                st.metric("Model", model_name)

            with result_col3:
                st.metric("Number of Cuisines", number_of_cuisines)

            st.progress(prediction / 5)

            # Bands follow the "Rating text" column in the dataset.
            if prediction >= 4.5:
                st.success("Excellent predicted rating")
            elif prediction >= 4.0:
                st.success("Very good predicted rating")
            elif prediction >= 3.5:
                st.info("Good predicted rating")
            elif prediction >= 2.5:
                st.warning("Average predicted rating")
            else:
                st.error("Poor predicted rating")

            st.subheader("Input features used by the model")

            full_row = pd.DataFrame(
                [
                    {
                        "Country": f"{COUNTRY_NAMES.get(country_code, 'Unknown')} ({country_code})",
                        "City": city,
                        "Cuisines": ", ".join(selected_cuisines),
                        "Average Cost for two": average_cost,
                        "Price range": price_range,
                        "Votes": votes,
                        "Has Table booking": table_booking,
                        "Has Online delivery": online_delivery,
                        "Number of Cuisines": number_of_cuisines,
                        "Latitude (city median)": round(latitude, 4),
                        "Longitude (city median)": round(longitude, 4),
                        "Model Used": model_name,
                        "Predicted Rating": round(prediction, 2),
                    }
                ]
            )

            summary_table = full_row.astype(str).T.rename(columns={0: "Value"})
            st.table(summary_table)

elif page == "Model Comparison":
    st.header("Model Comparison")

    st.caption(
        "The shared training utility uses a fixed 80/20 train-test split and "
        "5-fold cross-validation on the training 80%. The model is selected by "
        "lowest mean CV MAE; the held-out test set is used for final reporting. "
        "Production pipelines in the artifact are refit on all rated rows."
    )

    best_test = test_results[test_results["Model"] == best_model_name].iloc[0]

    metric_cols = st.columns(3)
    metric_cols[0].metric("Best Model (CV)", best_model_name)
    metric_cols[1].metric("Test MAE", f"{best_test['MAE']:.3f}")
    metric_cols[2].metric("Test R²", f"{best_test['R2']:.3f}")

    st.subheader("Cross-validation results (training set)")
    cv_display = cv_results.copy()
    for col in cv_display.columns[1:]:
        cv_display[col] = cv_display[col].round(4)
    st.dataframe(
        cv_display.rename(columns={"R2 mean": "R² mean", "R2 std": "R² std"}),
        width="stretch",
    )

    st.subheader("Hold-out test results")
    test_display = test_results.copy()
    for col in ["MAE", "RMSE", "R2"]:
        test_display[col] = test_display[col].round(4)
    st.dataframe(
        test_display.rename(columns={"R2": "R²"}),
        width="stretch",
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, metric in zip(axes, ["MAE mean", "RMSE mean", "R2 mean"]):
        ordered = cv_results.sort_values(metric, ascending=(metric != "R2 mean"))
        ax.bar(ordered["Model"], ordered[metric])
        ax.set_title(metric.replace("R2", "R²") + " (cross-validation)")
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle(
        "Model Comparison: lower MAE/RMSE and higher R² are better"
    )
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    final_estimator = best_model.named_steps["model"]
    encoded_names = best_model.named_steps["preprocessor"].get_feature_names_out()
    # Drop the "num__" / "cat__" prefixes added by the ColumnTransformer.
    encoded_names = [name.split("__", 1)[-1] for name in encoded_names]

    if hasattr(final_estimator, "feature_importances_"):
        values = final_estimator.feature_importances_
        title = f"{best_model_name} Feature Importance (top 15)"
        xlabel = "Importance"
    elif hasattr(final_estimator, "coef_"):
        values = np.ravel(final_estimator.coef_)
        title = f"{best_model_name} Coefficients (top 15)"
        xlabel = "Coefficient Value"
        st.caption(
            "Coefficients are not scaled, so their sizes cannot be compared across features."
        )
    else:
        values = None

    if values is not None:
        feature_data = pd.DataFrame(
            {
                "Feature": encoded_names,
                "Value": values,
            }
        )

        feature_data = (
            feature_data.sort_values("Value", key=abs, ascending=False)
            .head(15)
            .iloc[::-1]
        )

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(feature_data["Feature"], feature_data["Value"])
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Features (categories are one-hot encoded)")
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

elif page == "Dataset Overview":
    st.header("Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Original Rows", f"{len(raw_df):,}")
    c2.metric("Processed Rows", f"{len(df):,}")
    c3.metric("Columns", f"{len(raw_df.columns):,}")
    c4.metric(
        "Unrated Removed",
        f"{(raw_df[TARGET_COL] == 0).sum():,}",
    )

    rated_raw = raw_df[raw_df[TARGET_COL] > 0]
    placeholder_coord_count = int(
        ((rated_raw["Latitude"] == 0) | (rated_raw["Longitude"] == 0)).sum()
    )
    zero_cost_count = int((rated_raw["Average Cost for two"] == 0).sum())
    st.caption(
        f"Among the rated restaurants, {placeholder_coord_count} have placeholder "
        f"coordinates (0) and {zero_cost_count} have an average cost of 0. These "
        "are treated as missing values and filled before training."
    )

    st.subheader("Dataset Preview")
    st.dataframe(raw_df.head(10), width="stretch")

    st.subheader("Column Information")

    dtype_summary = pd.DataFrame(
        {
            "Column": raw_df.columns,
            "Data Type": raw_df.dtypes.astype(str).values,
            "Non-Null Count": raw_df.notna().sum().values,
            "Missing Count": raw_df.isna().sum().values,
            "Unique Values": raw_df.nunique().values,
        }
    )

    st.dataframe(dtype_summary, width="stretch")

    st.subheader("Numerical Statistics (rated restaurants, after cleaning)")

    numerical_cols = [
        "Average Cost for two",
        TARGET_COL,
        "Votes",
        "Price range",
    ]

    st.dataframe(
        df[numerical_cols].describe().round(2),
        width="stretch",
    )

    st.subheader("Feature Columns Used for Prediction")
    st.write("Numeric:", NUMERIC_FEATURES)
    st.write("Categorical (one-hot encoded):", CATEGORICAL_FEATURES)

else:
    st.header("Exploratory Data Analysis")

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Ratings",
            "Price & Delivery",
            "Cuisines",
            "Location & Correlation",
        ]
    )

    with tab1:
        st.subheader("Aggregate Rating Distribution")

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(
            df["Aggregate rating"],
            bins=20,
            kde=True,
            ax=ax,
        )
        ax.set_title("Distribution of Aggregate Rating")
        st.pyplot(fig)
        plt.close(fig)

        rating_counts = df["Rating text"].value_counts()

        fig, ax = plt.subplots(figsize=(8, 5))
        rating_counts.plot(kind="bar", ax=ax)
        ax.set_title("Rating Text Distribution")
        ax.set_xlabel("Rating Text")
        ax.set_ylabel("Number of Restaurants")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab2:
        st.subheader("Table Booking and Online Delivery")

        table_booking_dist = (
            df["Has Table booking"].value_counts(normalize=True) * 100
        )
        online_delivery_dist = (
            df["Has Online delivery"].value_counts(normalize=True) * 100
        )

        c1, c2 = st.columns(2)
        c1.metric(
            "Table Booking Available",
            f"{table_booking_dist.get('Yes', 0):.2f}%",
        )
        c2.metric(
            "Online Delivery Available",
            f"{online_delivery_dist.get('Yes', 0):.2f}%",
        )

        avg_rating_by_booking = df.groupby(
            "Has Table booking"
        )["Aggregate rating"].mean()

        fig, ax = plt.subplots(figsize=(7, 5))
        avg_rating_by_booking.plot(kind="bar", ax=ax)
        ax.set_title("Average Rating: Table Booking vs No Table Booking")
        ax.set_xlabel("Has Table Booking")
        ax.set_ylabel("Average Rating")
        ax.tick_params(axis="x", rotation=0)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        delivery_by_price_range = df.groupby(
            "Price range"
        )["Has Online delivery"].apply(
            lambda x: (x == "Yes").mean() * 100
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        delivery_by_price_range.plot(kind="bar", ax=ax)
        ax.set_title(
            "Percentage of Restaurants with Online Delivery by Price Range"
        )
        ax.set_xlabel("Price Range")
        ax.set_ylabel("Percentage Offering Online Delivery (%)")
        ax.set_ylim(0, 100)
        ax.tick_params(axis="x", rotation=0)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        avg_rating_by_price_range = df.groupby(
            "Price range"
        )["Aggregate rating"].mean()

        fig, ax = plt.subplots(figsize=(7, 5))
        avg_rating_by_price_range.plot(kind="bar", ax=ax)
        ax.set_title("Average Rating by Price Range")
        ax.set_xlabel("Price Range")
        ax.set_ylabel("Average Rating")
        ax.tick_params(axis="x", rotation=0)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab3:
        st.subheader("Cuisine Analysis")

        cuisine_df = df[
            ["Cuisines", "Aggregate rating", "Votes"]
        ].copy()

        cuisine_df["Cuisines"] = cuisine_df["Cuisines"].str.split(", ")
        cuisine_exploded = cuisine_df.explode("Cuisines")

        cuisine_votes = (
            cuisine_exploded.groupby("Cuisines")["Votes"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        cuisine_votes.plot(kind="bar", ax=ax)
        ax.set_title(
            "Top 10 Cuisines by Total Votes "
            "(Most Popular with Customers)"
        )
        ax.set_xlabel("Cuisine")
        ax.set_ylabel("Total Votes")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        cuisine_counts = cuisine_exploded["Cuisines"].value_counts()
        eligible_cuisines = cuisine_counts[cuisine_counts >= 20].index

        cuisine_avg_rating = (
            cuisine_exploded[
                cuisine_exploded["Cuisines"].isin(eligible_cuisines)
            ]
            .groupby("Cuisines")["Aggregate rating"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        cuisine_avg_rating.plot(kind="bar", ax=ax)
        ax.set_title(
            "Top 10 Highest-Rated Cuisines (min. 20 restaurants)"
        )
        ax.set_xlabel("Cuisine")
        ax.set_ylabel("Average Rating")
        ax.set_ylim(0, 5)
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.dataframe(
            cuisine_avg_rating.round(2).rename("Average Rating"),
            width="stretch",
        )

    with tab4:
        st.subheader("Restaurant Locations")

        fig, ax = plt.subplots(figsize=(9, 6))
        scatter = ax.scatter(
            df["Longitude"],
            df["Latitude"],
            c=df["Aggregate rating"],
            s=10,
            alpha=0.5,
        )
        ax.set_title("Restaurant Locations by Aggregate Rating")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(scatter, ax=ax, label="Aggregate Rating")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        corr_features = NUMERIC_FEATURES + [TARGET_COL]
        corr_matrix = df[corr_features].corr()

        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            ax=ax,
        )
        ax.set_title(
            "Correlation Between Numeric Features and Aggregate Rating"
        )
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        location_corr = df[
            ["Longitude", "Latitude", "Aggregate rating"]
        ].corr()

        st.subheader("Location Correlation")
        st.dataframe(location_corr.round(3), width="stretch")

st.sidebar.divider()
st.sidebar.caption(
    "Restaurant Rating Predictor"
)