import streamlit as st
import requests
import pandas as pd

# CONFIGURATION
API_URL = "https://shl-recommendation-pun8.onrender.com"

st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")

st.title("🤖 SHL Intelligent Assessment Recommender")
st.markdown("Enter a job description or skill requirement below to get AI-powered recommendations.")

# Input
query = st.text_area("Job Description / Query", height=100, placeholder="Example: Looking for a Java Developer who is also good at team collaboration...")

if st.button("Get Recommendations"):
    if not query.strip():
        st.warning("Please enter a query first.")
    else:
        with st.spinner("Analyzing requirements and searching catalog..."):
            try:
                response = requests.post(API_URL, json={"query": query})
                if response.status_code == 200:
                    data = response.json()
                    assessments = data.get("recommended_assessments", [])
                    
                    if not assessments:
                        st.info("No assessments found.")
                    else:
                        st.success(f"Found {len(assessments)} relevant assessments:")
                        
                        # Display as a clean table
                        df = pd.DataFrame(assessments)
                        # Reorder columns for display
                        display_cols = ["name", "test_type", "duration", "adaptive_support", "url"]
                        st.dataframe(
                            df[display_cols],
                            column_config={
                                "url": st.column_config.LinkColumn("SHL Link"),
                                "test_type": "Category"
                            },
                            hide_index=True
                        )
                        
                        # Detailed View
                        with st.expander("View Detailed Descriptions"):
                            for item in assessments:
                                st.markdown(f"### [{item['name']}]({item['url']})")
                                st.write(f"**Type:** {', '.join(item['test_type'])} | **Duration:** {item['duration']} mins")
                                st.write(item['description'])
                                st.divider()
                else:
                    st.error(f"API Error: {response.status_code}")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to API. Is 'python main.py' running?")