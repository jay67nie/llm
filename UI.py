import streamlit as st
# from chat_with_openai_assistant import chat_with_openai_assistant



def handle_user_question(user_question):
    print(user_question)
    st.write(st.session_state.messages)
    st.session_state.messages.append({"role":"user", "content":user_question})
    st.session_state.messages.append({"role":"assistant","content": "This is the response to the user question"})
    

    

def main():
     # Initialize session state for chat messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

   
    hide_decoration_bar_style = '''
        <style>
            header {visibility: hidden;}
        </style>
        '''
    st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)


    if user_question := st.chat_input("Ask Question here"):
        handle_user_question(user_question)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    


    


    with st.sidebar:
        st.write("This is sidebar that will contain the chat history")


if __name__ == "__main__":
    main()