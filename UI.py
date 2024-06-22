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

    options = ("Agriculture","Construction","Education","Health","Manufacturing","Retail and Wholesale")

    if st.session_state.sector is None:
        st.title("Welcome to URA Chatbot")
        sector = st.selectbox("Please select a sector", options)

        if st.button("Confirm Sector"):
            st.session_state.sector = sector
            with st.spinner("Setting sector ... Please Wait"):
                set_sector(sector)

    else:
        # Display header and sector selection on the sidebar
        with st.sidebar:
            col1,col2 = st.columns([1,3])
            with col1:
                st.image(ura_logo, width=50)
            with col2:
                st.title("URA Chatbot")
            sector = st.selectbox("Change Sector",options, index=options.index(st.session_state.sector))
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

if __name__ == "__main__":
    main()
