"""Point d'entrée de l'application Streamlit."""

import streamlit as st


def main() -> None:
    st.set_page_config(layout="wide", page_title="Strava — analyse d'activité")
    st.title("Analyse Strava")
    st.write("Sélectionnez une page dans le menu pour commencer.")


if __name__ == "__main__":
    main()
