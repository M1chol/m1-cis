import os
import sys
import traceback
from typing import List
import streamlit as st
from m1_cis import ContextSearch
from dotenv import load_dotenv

load_dotenv()

def get_keys():
    gemini = os.getenv("GEMINI_API_KEY")
    gkey = os.getenv("GOOGLE_API_KEY")
    gcx = os.getenv("GOOGLE_CX")
    missing = [k for k, v in [("GEMINI_KEY", gemini), ("GOOGLE_KEY", gkey), ("GOOGLE_CX", gcx)] if not v]
    if missing:
        st.warning(
            "Missing environment variables: "
            + ", ".join(missing)
            + ". Set them before running."
        )
    return gemini, gkey, gcx

def main():
    left, main, right = st.columns([1, 6, 1])
    with main:
        st.set_page_config(page_title="ContextSearch Demo v0.1.0", page_icon="🔎", layout="wide")
        st.title("🔎 ContextSearch Image Demo v0.1.0")
        st.write("Search for images and display their description and query.")

        gemini_key, google_key, google_cx = get_keys()
        query = st.text_input("Context:", value="", placeholder="Your context")
        custom_prompt = st.text_input("Custom prompt injection (optional), could be used to specify style", value="", placeholder="Custom injection")
        limit = st.number_input("Number of results", value=3)
        go = st.button("Search")

        if go:
            if not query.strip():
                st.info("Please enter a query.")
                return
            if not (gemini_key and google_key and google_cx):
                st.error("API keys are missing. Please set GEMINI_KEY, GOOGLE_KEY, GOOGLE_CX.")
                return

            with st.spinner("Running init..."):
                cs = ContextSearch(gemini_key, google_key, google_cx)

            with st.spinner("Searching for images..."):
                try:
                    images = cs.search(query, limit=limit, custom_prompt=custom_prompt)  # returns List[ImageSearchResult]
                except Exception as e:
                    st.error("Search failed.")
                    with st.expander("Details"):
                        st.code("".join(traceback.format_exception(*sys.exc_info())), language="text")
                    return

            if not images:
                st.warning("No results found.")
                return

            # Display results
            st.subheader("Results")
            for idx, img in enumerate(images, start=1):
                with st.container():
                    cols = st.columns([1, 2])
                    with cols[0]:
                        # Show image by URL
                        st.image(img.url, use_container_width=True, caption=f"Result {idx}")
                    with cols[1]:
                        st.markdown(f"• Description: {img.imageDescription}")
                        st.markdown(f"• Query: {img.imageSearchQuery}")
                        st.markdown(f"• URL: {img.url}")

            st.success(f"Loaded {len(images)} image(s).")


if __name__ == "__main__":
    main()