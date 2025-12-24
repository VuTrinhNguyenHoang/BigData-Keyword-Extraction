import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from services.backend.kw_methods import METHODS
import streamlit as st

st.set_page_config(
    page_title="Keyword Extraction",
    layout="wide",
    initial_sidebar_state="auto"
)

st.title("Keyword Extraction")

left_col, right_col = st.columns([7, 3])

with left_col:
    st.subheader("Enter the content from which you want to extract keywords:")

    # Input text
    text_input = st.text_area(
        "Input Text",
        height=300,
        placeholder="Type or paste your text here..."
    )

    # Method selection
    method = st.selectbox(
        "Select Keyword Extraction Method",
        options=list(METHODS.keys()),
        index=0
    )

    # Extract keywords button
    run_btn = st.button("Extract Keywords")

    if run_btn:
        if not text_input.strip():
            st.warning("Please enter some text to extract keywords.")
        else:
            extractor = METHODS.get(method)

            try:
                keywords = extractor(text_input, top_k=10)
            except Exception as e:
                st.error(f"An error occurred during keyword extraction: {e}")
                keywords = []
            
            st.session_state['keywords'] = keywords
            st.session_state['method'] = method

with right_col:
    st.subheader("Extracted Keywords:")

    if 'keywords' in st.session_state:
        st.write(f"**Method Used:** {st.session_state['method']}")
        kws = st.session_state['keywords']
        if not kws:
            st.write("No keywords were extracted.")
        else:
            for idx, kw in enumerate(kws, start=1):
                st.markdown(f"- **{idx}.** {kw}")
    else:
        st.write("No keywords extracted yet. Please enter text and click 'Extract Keywords'.")