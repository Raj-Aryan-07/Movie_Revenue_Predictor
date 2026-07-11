import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.sidebar.title("About the Developer")
st.sidebar.info("Built by Raj Aryan")
st.sidebar.markdown("[My LinkedIn Profile](https://www.linkedin.com/in/raj-aryan-298a99371/?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BARQLVDtEQNW4ElWOzFQhJQ%3D%3D)")
st.sidebar.markdown("[GitHub Repo : (https://github.com/Raj-Aryan-07/Movie_Revenue_Predictor) ]")

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(page_title="Movie Revenue Predictor", page_icon="🎬", layout="centered")

# Same 20 genres the model was trained on (matches df['genre'].nunique() == 20
# in the notebook, after "Science Fiction" was renamed to "Sci-Fi").
GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
    "Drama", "Family", "Fantasy", "Foreign", "History", "Horror", "Music",
    "Mystery", "Romance", "Sci-Fi", "TV Movie", "Thriller", "War", "Western",
]

CURRENT_YEAR = 2026


# ========================================================================
# >>> PUT / LOAD YOUR TRAINED MODEL HERE <<<
#
# This is the ONLY place you need to touch to plug in your own model.
# The rest of the app just calls `model.predict(input_df)`, so as long
# as `load_model()` returns an object with a `.predict()` method that
# accepts a DataFrame with columns ["genre", "year", "budget", "runtime"],
# everything else will work unchanged.
#
# Option A (default) — load the pickle file saved at the end of the
# notebook (`pickle.dump(pipe, file)`). Just drop the .pkl file next to
# this app.py:
#
# # This looks for the file in the exact same folder as your app.py
# # This works on your PC AND on the Streamlit Cloud server
# MODEL_PATH = Path(__file__).parent / "Movie_Revenue_Predictor.pkl"

# @st.cache_resource
# def load_model():
#     """Loads the pickled pipeline model."""
#     if not MODEL_PATH.exists():
#         st.error(f"Model file not found at: {MODEL_PATH}. Make sure it is in the same folder as app.py")
#         return None
#     try:
#         with open(MODEL_PATH, "rb") as f:
#             return pickle.load(f)
#     except Exception as e:
#         st.error(f"Error loading model: {e}")
#         return None

# # Initialize the model
# model = load_model()

#
# Option B — if you saved the model with joblib instead of pickle:
#
#     import joblib
#     MODEL_PATH = Path(__file__).parent / "Movie_Revenue_Predictor.joblib"
#
#     @st.cache_resource
#     def load_model():
#         return joblib.load(MODEL_PATH)
#
# Option C — if you want to point at a model stored somewhere else
# (e.g. a different folder, or a path you downloaded a cloud-hosted
# model to locally), just change the path:
#
#     MODEL_PATH = Path("/absolute/path/to/your/Movie_Revenue_Predictor.pkl")
#
# Whatever option you use, the model MUST be the full sklearn Pipeline
# (preprocessing + RandomForestRegressor) from the notebook, not just
# the bare RandomForestRegressor, because the app passes it raw
# genre/year/budget/runtime values and relies on the pipeline's
# ColumnTransformer to one-hot encode / scale them.
# ========================================================================

MODEL_PATH = Path(__file__).parent / "Movie_Revenue_Predictor.pkl"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


model = load_model()
# ========================================================================
# >>> END OF MODEL LOADING SECTION <<<
# ========================================================================


# ----------------------------------------------------------------------
# Session state / wizard setup
# ----------------------------------------------------------------------
TOTAL_STEPS = 5

if "step" not in st.session_state:
    st.session_state.step = 1
if "title" not in st.session_state:
    st.session_state.title = None
if "genre" not in st.session_state:
    st.session_state.genre = None
if "year" not in st.session_state:
    st.session_state.year = None
if "budget" not in st.session_state:
    st.session_state.budget = None
if "runtime" not in st.session_state:
    st.session_state.runtime = None


def go_next():
    st.session_state.step += 1


def go_back():
    st.session_state.step -= 1


def restart():
    st.session_state.step = 1
    st.session_state.title = None
    st.session_state.genre = None
    st.session_state.year = None
    st.session_state.budget = None
    st.session_state.runtime = None


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.title("Movie🎥🎬🎥 Revenue Predictor")
st.markdown("### Made by Raj Aryan with 🤞")
st.caption("Answer one question at a time and we'll estimate the box-office revenue.")
st.progress(min(st.session_state.step, TOTAL_STEPS) / TOTAL_STEPS)
st.divider()

# ----------------------------------------------------------------------
# Step 1: Movie title
# ----------------------------------------------------------------------
if st.session_state.step == 1:
    st.subheader("Step 1 of 5 — What's the movie name?")
    title = st.text_input("Movie title", value=st.session_state.title or "")

    if st.button("Next ➡️", type="primary", disabled=(len(title.strip()) == 0)):
        st.session_state.title = title.strip()
        go_next()
        st.rerun()

    if len(title.strip()) == 0:
        st.info("Enter a title to continue.")

# ----------------------------------------------------------------------
# Step 2: Genre
# ----------------------------------------------------------------------
elif st.session_state.step == 2:
    st.subheader("Step 2 of 5 — What's the movie's genre?")
    default_index = GENRES.index(st.session_state.genre) if st.session_state.genre in GENRES else 0
    genre = st.selectbox("Genre", GENRES, index=default_index)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back"):
            go_back()
            st.rerun()
    with col2:
        if st.button("Next ➡️", type="primary"):
            st.session_state.genre = genre
            go_next()
            st.rerun()

# ----------------------------------------------------------------------
# Step 3: Year
# ----------------------------------------------------------------------
elif st.session_state.step == 3:
    st.subheader("Step 3 of 5 — What year was (or will be) the movie released?")
    year = st.number_input(
        "Release year",
        min_value=1900,
        max_value=CURRENT_YEAR + 5,
        value=st.session_state.year or CURRENT_YEAR,
        step=1,
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back"):
            go_back()
            st.rerun()
    with col2:
        if st.button("Next ➡️", type="primary"):
            st.session_state.year = int(year)
            go_next()
            st.rerun()

# ----------------------------------------------------------------------
# Step 4: Budget
# ----------------------------------------------------------------------
elif st.session_state.step == 4:
    st.subheader("Step 4 of 5 — What's the movie's budget (in USD)?")
    budget = st.number_input(
        "Budget ($)",
        min_value=0,
        value=st.session_state.budget or 20_000_000,
        step=1_000_000,
        format="%d",
    )
    st.caption(f"≈ ${budget:,.00f}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back"):
            go_back()
            st.rerun()
    with col2:
        if st.button("Next ➡️", type="primary", disabled=(budget <= 0)):
            st.session_state.budget = float(budget)
            go_next()
            st.rerun()

    if budget <= 0:
        st.info("Budget must be greater than 0 to continue.")

# ----------------------------------------------------------------------
# Step 5: Runtime
# ----------------------------------------------------------------------
elif st.session_state.step == 5:
    st.subheader("Step 5 of 5 — What's the movie's runtime (in minutes)?")
    runtime = st.number_input(
        "Runtime (minutes)",
        min_value=1,
        max_value=400,
        value=st.session_state.runtime or 110,
        step=1,
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back"):
            go_back()
            st.rerun()
    with col2:
        if st.button("🔮 Predict Revenue", type="primary"):
            st.session_state.runtime = float(runtime)
            go_next()
            st.rerun()

# ----------------------------------------------------------------------
# Step 6: Result
# ----------------------------------------------------------------------
elif st.session_state.step == 6:
    st.subheader(f"🎉 Prediction Result for \"{st.session_state.title}\"")

    st.write("Here's what you told us:")
    summary = pd.DataFrame(
        {
            "Field": ["Title", "Genre", "Year", "Budget", "Runtime"],
            "Value": [
                st.session_state.title,
                st.session_state.genre,
                st.session_state.year,
                f"${st.session_state.budget:,.0f}",
                f"{st.session_state.runtime:.0f} min",
            ],
        }
    )
    st.table(summary.set_index("Field"))

    if model is None:
        st.error(
            f"Could not find the trained model file at `{MODEL_PATH.name}`. "
            "See the commented 'PUT / LOAD YOUR TRAINED MODEL HERE' section "
            "near the top of app.py for how to point the app at your model."
        )
    else:
        # NOTE: "title" is only used for display above — it is intentionally
        # NOT passed to the model, since the pipeline was only trained on
        # genre/year/budget/runtime.
        input_df = pd.DataFrame(
            {
                "genre": [st.session_state.genre],
                "year": [st.session_state.year],
                "budget": [st.session_state.budget],
                "runtime": [st.session_state.runtime],
            }
        )
        try:
            prediction = model.predict(input_df)[0]
            prediction = max(prediction, 0)
            st.success(f"### Estimated Revenue: ${prediction:,.0f}")
            st.balloons()

            if st.session_state.budget:
                roi = (prediction - st.session_state.budget) / st.session_state.budget * 100
                if roi >= 0:
                    st.write(f"That's a projected return on budget of **+{roi:,.1f}%**.")
                else:
                    st.write(f"That's a projected return on budget of **{roi:,.1f}%**.")
        except Exception as e:
            st.error(f"Something went wrong while predicting: {e}")

    st.divider()
    if st.button("🔁 Start Over"):
        restart()
        st.rerun()