import streamlit as st



def handle_user_question(user_question):
    print(user_question)

    

def main():
    hide_decoration_bar_style = '''
        <style>
            header {visibility: hidden;}
        </style>
        '''
    st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)

    user_question = st.chat_input("Ask Question here")

    if user_question:
        handle_user_question(user_question)
    with st.sidebar:
        st.write("This is sidebar that will contain the chat history")


if __name__ == "__main__":
    main()