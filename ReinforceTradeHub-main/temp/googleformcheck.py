import streamlit as st

def main():
    st.title("Streamlit Application with Google Form")

    # Session state to keep track of the form completion
    if 'form_completed' not in st.session_state:
        st.session_state.form_completed = False

    # Embed Google Form
    embed_code = '''<iframe src="https://docs.google.com/forms/d/e/1FAIpQLScmDESKWrouF0PgMO56SpYXvteasG5DUpQeUpeaR6NqbXvHew/viewform?usp=pp_url" width="700" height="520" frameborder="0" marginheight="0" marginwidth="0">Loading…</iframe>'''
    if not st.session_state.form_completed:
        st.markdown(embed_code, unsafe_allow_html=True)
        if st.button('I have completed the form'):
            st.session_state.form_completed = True

    # Check if form is completed to move to the next part of the application
    if st.session_state.form_completed:
        st.success("Thank you for filling out the form. Now moving to the next part.")
        # You can put your logic here for what happens next after the form
        # For example, moving to a different bucket
        if st.button('Go to Bucket 2'):
            # Logic for Bucket 2
            st.write("Welcome to Bucket 2!")

if __name__ == "__main__":
    main()
