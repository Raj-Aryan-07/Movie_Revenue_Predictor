# 🎬 Movie Revenue Predictor

A machine learning project that predicts a movie's box-office revenue based on its **genre**, **release year**, **production budget**, and **runtime**. The model is served through an interactive, step-by-step **Streamlit** web application.

---

## 📌 Project Overview

| Detail | Info |
|---|---|
| **Goal** | Predict box-office revenue for a movie |
| **Model Type** | Regression (Random Forest Regressor in an sklearn Pipeline) |
| **Preprocessing** | ColumnTransformer — One-Hot Encoding + Feature Scaling |
| **Frontend** | Streamlit web app (wizard-style, 5 input steps) |
| **Developer** | Raj Aryan |

---

## 📁 Project Structure

```
Project1/
├── Codes/
│   ├── Movie_Success_Predictor.ipynb   # EDA, feature engineering & model training
│   ├── Movie_Revenue_Predictor.pkl     # Trained scikit-learn pipeline (serialised with pickle)
│   ├── app.py                          # Streamlit web application
│   ├── requirements.txt                # Python dependencies
│   └── README.md                       # This file
│
└── Datasets/
    ├── tmdb_5000_movies.csv            # Primary TMDB movie dataset (~5.4 MB)
    └── movies_youtube_sentiments.csv   # YouTube trailer sentiment data (~319 KB)
```

---

## 🗂️ Datasets

| File | Size | Description |
|---|---|---|
| `tmdb_5000_movies.csv` | ~5.4 MB | Core movie metadata from TMDB — genres, budget, revenue, runtime, release date, etc. |
| `movies_youtube_sentiments.csv` | ~319 KB | YouTube trailer sentiment scores linked to movies |

> **Preprocessing note:** The genre `"Science Fiction"` from the TMDB dataset was renamed to `"Sci-Fi"` during feature engineering to keep genre labels concise and consistent.

---

## ⚙️ Features Used by the Model

The trained pipeline accepts four input features:

| Feature | Type | Description | Default in App |
|---|---|---|---|
| `genre` | Categorical | Primary genre (one of 20 supported values) | `Action` |
| `year` | Numerical | Release year of the movie | Current year (2026) |
| `budget` | Numerical | Production budget in USD | $20,000,000 |
| `runtime` | Numerical | Duration of the movie in minutes | 110 min |

> **Note:** Movie title is collected for display only and is **not** passed to the model.

### Supported Genres (20 total)

`Action` · `Adventure` · `Animation` · `Comedy` · `Crime` · `Documentary` · `Drama` · `Family` · `Fantasy` · `Foreign` · `History` · `Horror` · `Music` · `Mystery` · `Romance` · `Sci-Fi` · `TV Movie` · `Thriller` · `War` · `Western`

---

## 🧠 Model Pipeline

The saved model (`Movie_Revenue_Predictor.pkl`) is a full **scikit-learn `Pipeline`** — not a bare regressor:

```
Pipeline
 ├── ColumnTransformer
 │    ├── OneHotEncoder  ──▶  genre  (categorical)
 │    └── StandardScaler ──▶  year, budget, runtime  (numerical)
 └── RandomForestRegressor  ──▶  revenue prediction (USD)
```

> ⚠️ **Important:** The app passes raw (unencoded) input directly to `model.predict()`. The full Pipeline **must** be used — replacing it with only the bare `RandomForestRegressor` will break the app, since encoding and scaling happen inside the Pipeline.

The model is loaded once with `@st.cache_resource`, so it is **not** reloaded on every user interaction — keeping the app fast and memory-efficient.

---

## 🚀 Getting Started

### 1. Clone / Download the Project

```bash
git clone <your-repo-url>
cd "ML Projects/Project1/Codes"
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Web App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 🖥️ Streamlit App — How It Works

The app walks you through a **5-step input wizard**, then shows a prediction result:

| Step | Input | Validation |
|---|---|---|
| 1 | 🎬 Movie Title | Must be non-empty |
| 2 | 🎭 Genre | Select from 20 genres |
| 3 | 📅 Release Year | Range: 1900 – 2031 |
| 4 | 💰 Production Budget (USD) | Must be greater than $0 |
| 5 | ⏱️ Runtime (minutes) | Range: 1 – 400 min |

### Result Page

After completing all steps the app:
- Displays a **summary table** of your inputs
- Runs the trained pipeline and shows the **estimated box-office revenue**
- Calculates a projected **Return on Budget (ROI)**:
  ```
  ROI (%) = (Predicted Revenue − Budget) / Budget × 100
  ```
- Launches a 🎈 balloon animation on a successful prediction
- Offers a **"Start Over"** button to reset the wizard

### Sidebar

The app sidebar (visible at all times) shows:
- Developer name: **Raj Aryan**
- 🔗 [LinkedIn Profile](https://www.linkedin.com/in/raj-aryan-298a99371/)
- 🔗 [GitHub Repo Link](https://github.com/Raj-Aryan-07/Movie_Revenue_Predictor)

---

## 📦 Requirements

```
streamlit
pandas
numpy
scikit-learn
```

Install with:

```bash
pip install -r requirements.txt
```

---

## 📓 Notebook — `Movie_Success_Predictor.ipynb`

The notebook covers the complete ML workflow:

1. **Exploratory Data Analysis (EDA)** — understanding distributions, correlations, and outliers
2. **Data Cleaning** — handling missing values and inconsistent entries
3. **Feature Engineering** — genre extraction, date parsing, genre renaming (`Science Fiction` → `Sci-Fi`)
4. **Pipeline Construction** — `ColumnTransformer` for encoding/scaling + `RandomForestRegressor`
5. **Model Training & Evaluation** — fitting on TMDB data, performance metrics
6. **Model Export** — serialising the full pipeline to `Movie_Revenue_Predictor.pkl` via `pickle.dump(pipe, file)`

---

## 📉 Model Performance & Limitations

The model produces **reasonable directional estimates** but has a **low R² score** — meaning it explains only a modest portion of the variance in actual box-office revenue. This is expected and well-documented in the movie revenue prediction literature.

### Why the R² score is low

| Reason | Explanation |
|---|---|
| **Too few features Used** | The model only sees 4 inputs: genre, year, budget, and runtime. Movie success is driven by dozens of other factors not captured here. |
| **Missing critical signals** | Star power (cast & director), marketing spend, critic/audience reviews, social media buzz, and competition at release date are all absent. |
| **Highly skewed revenue distribution** | A handful of blockbusters (Marvel, Avatar, etc.) earn 10–100× the average. This extreme variance makes accurate regression very hard. |
| **Noisy dataset** | The TMDB dataset contains many entries where budget or revenue is listed as `0` (unreported), adding significant noise to training. |
| **Inherent unpredictability** | Even Hollywood studios with full production data and experienced analysts routinely mis-forecast box-office performance. Revenue is partly driven by cultural moments and timing that no model can quantify. |
| **No log-transformation on target** | Revenue spans several orders of magnitude ($10K → $2B+). Without log-scaling the target variable, the model is penalised heavily by outlier blockbusters. |

### What the model is still useful for

- **Ballpark estimates** — gives a reasonable order-of-magnitude for expected revenue
- **Relative comparisons** — comparing two hypothetical movies under the same conditions
- **Educational demonstration** — end-to-end ML pipeline: data → model → deployed web app

> 💡 **Improving the model:** Better performance could be achieved by adding cast popularity scores, director track record, trailer sentiment (the `movies_youtube_sentiments.csv` dataset is already available!), marketing budget, and applying a log transform to the revenue target before training.

---

## 🔧 Customising the Model

To swap in your own model, edit the **"PUT / LOAD YOUR TRAINED MODEL HERE"** section near the top of `app.py`. The app supports:

| Option | Method |
|---|---|
| **Option A** (default) | Load `.pkl` file via `pickle` — file must be in the same folder as `app.py` |
| **Option B** | Load `.joblib` file via `joblib.load()` |
| **Option C** | Point to an absolute path anywhere on disk or a downloaded cloud model |

The only contract: the loaded object must have a `.predict(DataFrame)` method that accepts columns `["genre", "year", "budget", "runtime"]`.

---

## 👤 Developer

**Raj Aryan**
- 🔗 [LinkedIn Profile](https://www.linkedin.com/in/raj-aryan-298a99371/)
- 🔗 [GitHub Repo Link](https://github.com/Raj-Aryan-07/Movie_Revenue_Predictor)
