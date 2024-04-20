import streamlit as st


def check_text_tone_tag(text):
    return "Tested tone tag"


def main():
    st.header(f"Check tone tag of your message/text.")

    max_chars = 5000

    users_text = st.text_input("Write your text here:", max_chars=max_chars)
    remaining_chars = max_chars - len(users_text)
    st.write(f"Chars left:{remaining_chars}")

    if st.button("Run it"):
        tonetag = check_text_tone_tag(users_text)
        st.write(f"Tone Tag: {tonetag}")


if __name__ == "__main__":
    main()
