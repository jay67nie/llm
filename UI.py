import streamlit as st
from chat_with_openai_assistant_evaluate import chat_with_assistant, set_sector
from PIL import Image

# Load URA logo image
ura_logo = Image.open('./images/ura_logo.jpeg')

# Set page configuration
st.set_page_config(page_title="URA Chatbot", page_icon=ura_logo)


# Function to handle user question
def handle_user_question(user_question):
    st.session_state.awaiting_response = True  # Set a flag to indicate response is being generated

    # Fetch the response
    response = chat_with_assistant(user_question)

    # Append the response to the messages list
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.awaiting_response = False  # Reset the flag
    st.rerun()  # Rerun the app to update the UI


# Function to reset sector
def reset_sector(new_sector):
    # Reset to the new sector
    st.session_state.sector = new_sector
    set_sector(new_sector)
    st.session_state.messages = []
    st.session_state.awaiting_response = False


# Main function
def main():
    # Initialize session state for chat messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize session state for sector
    if "sector" not in st.session_state:
        st.session_state.sector = None

    # Initialize session state for awaiting response
    if "awaiting_response" not in st.session_state:
        st.session_state.awaiting_response = False

    # Hide decoration bar style
    hide_decoration_bar_style = '''
        <style>
            header {visibility: hidden;}
        </style>
        '''
    st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)

    options = ("Agriculture", "Construction", "Education", "Health", "Manufacturing", "Retail and Wholesale")

    if st.session_state.sector is None:
        # Display header and sector selection
        st.markdown("""
        <div style='display: flex; align-items: center;'>
            <img src='data:image/png;base64,{0}' class='img-fluid' style='margin-right: 10px; width: 48px; height: auto;
            '>
            <h1>Welcome to the URA Chatbot</h1>
        </div>
    """.format(image_to_base64(ura_logo)), unsafe_allow_html=True)

        # Display sector selection
        sector = st.selectbox("Please select a sector", options)

        if st.button("Confirm Sector"):  # Set the sector
            st.session_state.sector = sector
            with st.spinner("Setting sector ... Please Wait"):
                set_sector(sector)
                st.session_state.messages = [{
                    "role": "assistant",
                    "content": f"Hello there, you can ask me any question with regards to taxes in the {sector}"
                               f" sector! I shall try my best to answer them. However, "
                               f"remember that AI can also make mistakes. "
                               f"Please check important information."}]
                st.rerun()

    else:
        # Display header and sector selection on the sidebar
        with st.sidebar:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(ura_logo, width=50)
            with col2:
                st.title("URA Chatbot")
            sector = st.selectbox("Change Sector", options, index=options.index(st.session_state.sector))
            if st.button("Change Sector"):
                reset_sector(sector)

        user_question = st.chat_input("Ask Question here", disabled=st.session_state.awaiting_response)

        # Check if there is a user question and handle it
        if user_question:
            st.session_state.messages.append({"role": "user", "content": user_question})
            st.session_state.awaiting_response = True  # Set the flag to indicate the response is being generated

        # Display chat messages
        for message in st.session_state.messages:
            if message["role"] == "assistant":
                col1, col2 = st.columns([1, 20])
                with col1:
                    st.image(ura_logo, width=25)
                with col2:
                    st.write(message["content"])
            else:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

        if st.session_state.awaiting_response:
            with st.spinner("Generating response..."):
                handle_user_question(st.session_state.messages[-1]["content"])

            # Function to convert image to base64


def image_to_base64(image_path):
    import base64
    from io import BytesIO

    buffered = BytesIO()
    image_path.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


if __name__ == "__main__":
    main()
