import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

st.set_page_config(
    page_title="Restaurant Rating Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

sns.set_style("whitegrid")

FEATURE_COLS = [
    "Average Cost for two",
    "Price range",
    "Votes",
    "Table Booking Binary",
    "Online Delivery Binary",
    "Number of Cuisines",
    "City Encoded",
    "Country Code Encoded",
]

COUNTRY_NAMES = {
    1: "India", 14: "Australia", 30: "Brazil", 37: "Canada",
    94: "Indonesia", 148: "New Zealand", 162: "Philippines",
    166: "Qatar", 184: "Singapore", 189: "South Africa",
    191: "Sri Lanka", 208: "Turkey", 214: "UAE",
    215: "United Kingdom", 216: "United States",
}

REQUIRED_COLUMNS = [
    "Restaurant ID",
    "Restaurant Name",
    "Country Code",
    "City",
    "Address",
    "Locality",
    "Locality Verbose",
    "Longitude",
    "Latitude",
    "Cuisines",
    "Average Cost for two",
    "Currency",
    "Has Table booking",
    "Has Online delivery",
    "Is delivering now",
    "Switch to order menu",
    "Price range",
    "Aggregate rating",
    "Rating color",
    "Rating text",
    "Votes",
]

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
    data = pd.read_csv("Dataset.csv")

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in data.columns]
    if missing_columns:
        raise ValueError(
            "The dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    return data


@st.cache_data
def preprocess_data(raw_df):
    df = raw_df.copy()

    df["Cuisines"] = df["Cuisines"].fillna("Unknown")

    df = df[df["Aggregate rating"] > 0].copy()

    df["Restaurant Name Length"] = df["Restaurant Name"].apply(len)
    df["Address Length"] = df["Address"].apply(len)

    df["Table Booking Binary"] = (
        df["Has Table booking"] == "Yes"
    ).astype(int)

    df["Online Delivery Binary"] = (
        df["Has Online delivery"] == "Yes"
    ).astype(int)

    df["Number of Cuisines"] = df["Cuisines"].apply(
        lambda x: len(str(x).split(", "))
    )

    city_freq_map = df["City"].value_counts().to_dict()
    df["City Encoded"] = df["City"].map(city_freq_map)

    country_freq_map = df["Country Code"].value_counts().to_dict()
    df["Country Code Encoded"] = df["Country Code"].map(country_freq_map)

    return df, city_freq_map, country_freq_map


@st.cache_data
def get_unique_cuisines(df, country_code=None):
    subset = df if country_code is None else df[df["Country Code"] == country_code]
    exploded = subset["Cuisines"].dropna().str.split(", ").explode().str.strip()
    return sorted(exploded[exploded != ""].unique().tolist())

@st.cache_resource
def train_models(df):
    X = df[FEATURE_COLS]
    y = df["Aggregate rating"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42),
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        results.append(
            {
                "Model": name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
            }
        )

    results_df = (
        pd.DataFrame(results)
        .sort_values(by="R2", ascending=False)
        .reset_index(drop=True)
    )

    best_model_name = results_df.iloc[0]["Model"]
    best_model = models[best_model_name]

    return (
        models,
        results_df,
        best_model_name,
        best_model,
        X_train,
        X_test,
        y_train,
        y_test,
    )

st.markdown(
    '<div class="main-title">Restaurant Rating Predictor</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">'
    "Interactive Streamlit version of the restaurant prediction notebook"
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
    df, city_freq_map, country_freq_map = preprocess_data(raw_df)

    (
        models,
        results_df,
        best_model_name,
        best_model,
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_models(df)

except FileNotFoundError:
    st.error(
        "Dataset.csv was not found. Place Dataset.csv in the same folder as app.py."
    )
    st.stop()

except ValueError as exc:
    st.error(str(exc))
    st.stop()

if page == "Prediction":
    st.header("Predict Restaurant Rating")

    st.info(
        f"Best model from the notebook workflow: **{best_model_name}** "
        f"(highest test-set R²)."
    )

    col1, col2 = st.columns(2)

    with col1:
        average_cost = st.number_input(
            "Average Cost for two",
            min_value=0.0,
            value=float(df["Average Cost for two"].median()),
            step=10.0,
        )

        price_range = st.selectbox(
            "Price range",
            sorted(df["Price range"].dropna().unique().tolist()),
            index=1 if len(df["Price range"].unique()) > 1 else 0,
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
        default_cuisine = ["North Indian"] if "North Indian" in cuisine_options else cuisine_options[:1]
        selected_cuisines = st.multiselect(
            "Cuisines",
            cuisine_options,
            default=default_cuisine,
            help="Only cuisines actually served in the selected country are shown.",
        )

        city_options = sorted(
            df.loc[df["Country Code"] == country_code, "City"].dropna().unique().tolist()
        )
        city = st.selectbox("City", city_options)

    model_name = st.selectbox(
        "Prediction model",
        list(models.keys()),
        index=list(models.keys()).index(best_model_name),
    )

    number_of_cuisines = len(selected_cuisines)

    input_data = pd.DataFrame(
        [
            {
                "Average Cost for two": average_cost,
                "Price range": price_range,
                "Votes": votes,
                "Table Booking Binary": int(table_booking == "Yes"),
                "Online Delivery Binary": int(online_delivery == "Yes"),
                "Number of Cuisines": number_of_cuisines,
                "City Encoded": city_freq_map.get(city, 0),
                "Country Code Encoded": country_freq_map.get(country_code, 0),
            }
        ]
    )

    if st.button("Predict Rating", type="primary", use_container_width=True):
        if not selected_cuisines:
            st.warning("Please select at least one cuisine before predicting.")
        else:
            selected_model = models[model_name]
            prediction = float(selected_model.predict(input_data)[0])

            # Ratings in the source data are on a 0–5 scale.
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

            if prediction >= 4.5:
                st.success("Excellent predicted rating")
            elif prediction >= 4.0:
                st.success("Very good predicted rating")
            elif prediction >= 3.0:
                st.info("Average/good predicted rating")
            elif prediction >= 2.0:
                st.warning("Below-average predicted rating")
            else:
                st.error("Low predicted rating")

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
                        "City Encoded": city_freq_map.get(city, 0),
                        "Country Code Encoded": country_freq_map.get(country_code, 0),
                        "Model Used": model_name,
                        "Predicted Rating": round(prediction, 2),
                    }
                ]
            )

            st.table(full_row.T.rename(columns={0: "Value"}))

elif page == "Model Comparison":
    st.header("Model Comparison")

    st.caption(
        "Models, train/test split, metrics, and selection follow the notebook."
    )

    metric_cols = st.columns(3)

    metric_cols[0].metric(
        "Best Model",
        best_model_name,
    )
    metric_cols[1].metric(
        "Best R²",
        f"{results_df.iloc[0]['R2']:.3f}",
    )
    metric_cols[2].metric(
        "Best RMSE",
        f"{results_df.iloc[0]['RMSE']:.3f}",
    )

    display_results = results_df.copy()
    for col in ["MAE", "RMSE", "R2"]:
        display_results[col] = display_results[col].round(3)

    st.dataframe(display_results, use_container_width=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, metric in zip(axes, ["MAE", "RMSE", "R2"]):
        ax.bar(results_df["Model"], results_df[metric])
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle(
        "Model Comparison: Lower MAE/RMSE and Higher R² = Better"
    )
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    if hasattr(best_model, "feature_importances_"):
        values = best_model.feature_importances_
        title = f"{best_model_name} Feature Importance"
        xlabel = "Importance"
    elif hasattr(best_model, "coef_"):
        values = best_model.coef_
        title = f"{best_model_name} Coefficients"
        xlabel = "Coefficient Value"
    else:
        values = None

    if values is not None:
        feature_data = pd.DataFrame(
            {
                "Feature": FEATURE_COLS,
                "Value": values,
            }
        )

        feature_data = feature_data.sort_values(
            "Value",
            key=abs,
            ascending=True,
        )

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(feature_data["Feature"], feature_data["Value"])
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Features")
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
        f"{(raw_df['Aggregate rating'] == 0).sum():,}",
    )

    st.subheader("Dataset Preview")
    st.dataframe(raw_df.head(10), use_container_width=True)

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

    st.dataframe(dtype_summary, use_container_width=True)

    st.subheader("Numerical Statistics")

    numerical_cols = [
        "Average Cost for two",
        "Aggregate rating",
        "Votes",
        "Price range",
    ]

    st.dataframe(
        df[numerical_cols].describe().round(2),
        use_container_width=True,
    )

    st.subheader("Feature Columns Used for Prediction")
    st.write(FEATURE_COLS)

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
            use_container_width=True,
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

        corr_features = FEATURE_COLS + ["Aggregate rating"]
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
            "Correlation Between Features and Aggregate Rating"
        )
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        location_corr = df[
            ["Longitude", "Latitude", "Aggregate rating"]
        ].corr()

        st.subheader("Location Correlation")
        st.dataframe(location_corr.round(3), use_container_width=True)

st.sidebar.divider()
st.sidebar.caption(
    "Restaurant Rating Predictor • Based on restaurant_predict.ipynb"
)