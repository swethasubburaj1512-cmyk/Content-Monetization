# streamlit_youtube_revenue.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from googleapiclient.discovery import build
import datetime
import re

# --- Paths for model artifacts ---
ARTIFACT_DIR = "./model_artifacts"
PIPELINE_PATH = os.path.join(ARTIFACT_DIR, "LinearRegression.pkl")
COLUMNS_PATH = os.path.join(ARTIFACT_DIR, "model_input_columns.pkl")

# --- Streamlit page config ---
st.set_page_config(
    page_title="YouTube Ad Revenue Predictor",
    layout="wide",
    page_icon="https://upload.wikimedia.org/wikipedia/commons/b/b8/YouTube_Logo_2017.svg"
)

# --- Header with YouTube logo ---
st.markdown(
    """
    <div style='display:flex; justify-content: space-between; align-items: center;'>
        <h1 style='color: #FF0000;'>YouTube Ad Revenue Predictor</h1>
        <img src='https://upload.wikimedia.org/wikipedia/commons/b/b8/YouTube_Logo_2017.svg' width='80'>
    </div>
    <hr style='border:1px solid #FF0000'>
    """,
    unsafe_allow_html=True
)

# --- Load model ---
if not os.path.exists(PIPELINE_PATH) or not os.path.exists(COLUMNS_PATH):
    st.error("Model artifacts not found. Run train_model.py first.")
    st.stop()

pipeline = joblib.load(PIPELINE_PATH)
required_columns = joblib.load(COLUMNS_PATH)

# --- YouTube API ---
API_KEY = "AIzaSyDUz8lluwwq_cRbPqYqzwLTCAnJ_aKtNdg"  

def parse_duration(duration):
    """Convert ISO 8601 duration (PT#H#M#S) to seconds"""
    hours = minutes = seconds = 0
    match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration)
    if match:
        if match.group(1): hours = int(match.group(1))
        if match.group(2): minutes = int(match.group(2))
        if match.group(3): seconds = int(match.group(3))
    return hours*3600 + minutes*60 + seconds

def fetch_youtube_metadata_api(video_url):
    """Fetch video metadata using YouTube Data API"""
    try:
        # Extract video ID
        if "v=" in video_url:
            video_id = video_url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in video_url:
            video_id = video_url.split("youtu.be/")[1].split("?")[0]
        else:
            return {"error": "Invalid YouTube URL"}

        youtube = build("youtube", "v3", developerKey=API_KEY)
        response = youtube.videos().list(
            part="snippet,statistics,contentDetails",
            id=video_id
        ).execute()

        if not response['items']:
            return {"error": "Video not found or private"}

        item = response['items'][0]
        snippet = item['snippet']
        stats = item.get('statistics', {})
        content = item['contentDetails']
        publish_date = datetime.datetime.strptime(snippet['publishedAt'], "%Y-%m-%dT%H:%M:%SZ")

        metadata = {
            "title": snippet.get("title", ""),
            "views": int(stats.get("viewCount", 0)),
            "likes": int(stats.get("likeCount", 0)),
            "comments": int(stats.get("commentCount", 0)),
            "subscribers": 0,  # API v3 needs separate channel API call
            "publish_date": publish_date,
            "video_length_seconds": parse_duration(content.get("duration", "PT0S")),
            "thumbnail_url": snippet['thumbnails']['high']['url']
        }
        metadata["video_length_minutes"] = metadata["video_length_seconds"] / 60.0
        return metadata

    except Exception as e:
        return {"error": str(e)}

# --- Sidebar input method ---
st.sidebar.header("Input method")
input_mode = st.sidebar.radio("Choose how to provide video info:", ["Paste YouTube link (recommended)", "Manual input"])

st.write("The model predicts **ad revenue (USD)** for a YouTube video based on metadata and engagement.")

# --- Input capture ---
meta = None
url = None
if input_mode == "Paste YouTube link (recommended)":
    url = st.text_input("Paste YouTube video URL here")
    if st.button("Fetch metadata"):
        if not url:
            st.warning("Please paste a YouTube URL.")
        else:
            with st.spinner("Fetching metadata..."):
                meta = fetch_youtube_metadata_api(url)
            if meta and "error" not in meta:
                st.success("Metadata fetched.")
                st.markdown(f"**Title:** {meta.get('title')}")
                st.markdown(f"**Views:** {meta.get('views'):,}")
                st.markdown(f"**Video length (min):** {meta.get('video_length_minutes'):.2f}")
                st.image(meta["thumbnail_url"], caption="Video thumbnail", use_column_width=True)
            else:
                st.error(meta.get("error", "Unknown error"))

# --- Manual / fallback inputs ---
st.subheader("Video & engagement details")
inputs = {}
col1, col2 = st.columns(2)
with col1:
    inputs['views'] = st.number_input("Views", min_value=0, value=0, step=1)
    inputs['likes'] = st.number_input("Likes", min_value=0, value=0, step=1)
    inputs['comments'] = st.number_input("Comments", min_value=0, value=0, step=1)
with col2:
    inputs['watch_time_minutes'] = st.number_input("Watch time (minutes)", min_value=0.0, value=0.0, step=1.0)
    inputs['video_length_minutes'] = st.number_input("Video length (minutes)", min_value=0.0, value=0.0, step=0.5)
    inputs['subscribers'] = st.number_input("Channel subscribers", min_value=0, value=0, step=100)

# Context
st.subheader("Context (categorical)")
inputs['category'] = st.text_input("Category (e.g., Entertainment, Education, Music)", value="missing")
inputs['device'] = st.selectbox("Device", options=["mobile", "desktop", "tablet", "other"], index=0)
inputs['country'] = st.text_input("Country (ISO or name)", value="missing")

# --- Fill inputs from metadata ---
if meta and "error" not in meta:
    if inputs['views'] == 0: inputs['views'] = meta.get('views',0)
    if inputs['likes'] == 0: inputs['likes'] = meta.get('likes',0)
    if inputs['comments'] == 0: inputs['comments'] = meta.get('comments',0)
    if inputs['video_length_minutes'] == 0: inputs['video_length_minutes'] = meta.get('video_length_minutes',0)
    publish_date = meta.get('publish_date')
    if publish_date:
        inputs['upload_month'] = publish_date.month
        inputs['upload_dayofweek'] = publish_date.weekday()
else:
    if 'upload_month' not in inputs:
        inputs['upload_month'] = st.number_input("Upload month (1-12, 0 if unknown)", min_value=0, max_value=12, value=0, step=1)
    if 'upload_dayofweek' not in inputs:
        inputs['upload_dayofweek'] = st.number_input("Upload day of week (0=Mon..6=Sun, 0 if unknown)", min_value=0, max_value=6, value=0, step=1)

# --- Build DataFrame ---
def build_input_df(inputs, required_cols):
    d = {}
    numeric_cols = ['views','likes','comments','watch_time_minutes','video_length_minutes','subscribers','upload_month','upload_dayofweek']
    cat_cols = ['category','device','country']
    for c in required_cols: d[c]=0
    for col in numeric_cols:
        if col in required_cols: d[col]=float(inputs.get(col,0))
    for col in cat_cols:
        if col in required_cols: d[col]=inputs.get(col,"missing")
    # engineered features
    try: d['log_views']=np.log1p(max(0,d['views'])); 
    except: d['log_views']=0
    try: d['log_likes']=np.log1p(max(0,d['likes'])); 
    except: d['log_likes']=0
    try: d['log_comments']=np.log1p(max(0,d['comments'])); 
    except: d['log_comments']=0
    try: d['log_watch_time_minutes']=np.log1p(max(0,d['watch_time_minutes']));
    except: d['log_watch_time_minutes']=0
    views=max(1.0,d['views'])
    d['engagement_rate']=(d['likes']+d['comments'])/views
    d['watch_time_per_view']=d['watch_time_minutes']/views
    return pd.DataFrame([d])

# --- Predict ---
if st.button("Predict ad revenue (USD)"):
    input_df = build_input_df(inputs, required_columns)
    st.write("Model input preview:")
    st.dataframe(input_df.T.rename(columns={0:"value"}))
    try:
        prediction = pipeline.predict(input_df)[0]
        st.success(f"Predicted ad revenue (USD): {prediction:.2f}")
        st.info("This is an estimate. Manual input overrides API values if changed.")
    except Exception as e:
        st.error("Prediction failed: " + str(e))

st.markdown("---")
st.caption("The app uses model artifacts from train_model.py. If features change, retrain the model.")
