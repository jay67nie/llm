import streamlit as st
from chat_with_openai_assistant_evaluate import chat_with_assistant, set_sector
from PIL import Image


ura_logo = Image.open('./images/ura_logo.jpeg')

st.set_page_config(page_title="URA Chatbot", page_icon=ura_logo)



def handle_user_question(user_question):
    st.session_state.awaiting_response = True  # Set a flag to indicate response is being generated

    # Fetch the response
    response = chat_with_assistant(user_question)

    st.session_state.messages.append({"role": "assistant", "content": response}) # Append the response to the messages list
    st.session_state.awaiting_response = False  # Reset the flag
    st.rerun()  # Rerun the app to update the UI

def reset_sector(new_sector):
    # Reset to the new sector
    st.session_state.sector = new_sector
    set_sector(new_sector)
    st.session_state.messages = []
    st.session_state.awaiting_response = False
    # st.rerun()
    

    

def main():
     # Initialize session state for chat messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "sector" not in st.session_state:
        st.session_state.sector = None

    if "awaiting_response" not in st.session_state: 
        st.session_state.awaiting_response = False
    

   
    hide_decoration_bar_style = '''
        <style>
            header {visibility: hidden;}
        </style>
        '''
    st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)

    options = ("Agriculture","Construction","Education","Health","Manufacturing","Retail and Wholesale");

    if st.session_state.sector is None:
        st.title("Welcome to URA Chatbot")
        sector = st.selectbox("Please select a sector", options)

        if st.button("Confirm Sector"):
            st.session_state.sector = sector
            with st.spinner("Setting sector ... Please Wait"):
                set_sector(sector)

    else:
        # on the side bar, display header and sector selection
        with st.sidebar:
            st.title("URA Chatbot")
            sector = st.selectbox("Change Sector",options, index=options.index(st.session_state.sector))
            if st.button("Change Sector"):
                reset_sector(sector)

        user_question = st.chat_input("Ask Question here", disabled=st.session_state.awaiting_response)
        # st.caption("AI can also make mistakes. Check Important information.")

        # Check if there is a user question and handle it
        if user_question:    
        # if user_question := st.chat_input("Ask Question here", disabled=st.session_state.awaiting_response):
            st.session_state.messages.append({"role": "user", "content": user_question})
            st.session_state.awaiting_response = True  # Set the flag to indicate the response is being generated
        
        # st.markdown("""
        # <div style="vertical-align: top;'>
        # <small>Welcome to My App</small>
        # </div>
        # """, unsafe_allow_html=True)


        for message in st.session_state.messages:
            if message["role"] == "assistant":
                col1, col2 = st.columns([1, 20])
                with col1:
                    st.image(ura_logo, width=25)  # Adjust the width as needed
                with col2:
                    st.write(message["content"])
            else:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

            

        if st.session_state.awaiting_response:
            with st.spinner("Generating response..."):
                handle_user_question(st.session_state.messages[-1]["content"]) 

                

# TODO add streaming and changing the sector


if __name__ == "__main__":
    main()
