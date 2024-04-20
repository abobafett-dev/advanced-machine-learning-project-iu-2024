import streamlit as st


def check_text_tone_tag(text):
    return "Tested tone tag"


def main():
    st.header(f"Check tone tag of your message/text")

    users_text = st.text_area("Write your text here:", height=500, max_chars=500000)

    if st.button("Run it"):
        tone_tag = check_text_tone_tag(users_text)
        st.write(f"{tone_tag}")


if __name__ == "__main__":
    main()
