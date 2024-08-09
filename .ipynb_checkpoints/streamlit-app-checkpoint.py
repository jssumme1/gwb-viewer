import streamlit as st

st.set_page_config(layout="centered",
                   page_title="Gravitational-Wave Background Visualizer",
                   page_icon="https://gwosc.org/static/images/icons/gwosc-fav.ico",
                   menu_items = { "Get help": "https://ask.igwn.org"})

st.title('Gravitational-Wave Background Visualizer')

st.markdown("""Visualize the stochastic gravitational-wave background that various populations of stellar-mass binary black holes would create.
""")

st.markdown("""
 * Use the menu on the left to select the merger rate for various bins of redshift.
 * Then also use the menu to select the shape of the mass distribution at each redshift bin.
 * Your plots will appear below.
""")
