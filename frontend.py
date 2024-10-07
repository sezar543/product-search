
import os
from dotenv import find_dotenv, load_dotenv
import requests
from QA_WCD_chatbot_Streamlit_2 import Setup_llm, qa_ENSEM_Run


import streamlit as st
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables.config import RunnableConfig


import asyncio

backend_url = "http://localhost:8000/chatbot"

def store_userinfo(user_contact):
    # Check if 'feedback' already exists in session_state
    # If not, then initialize it as an empty list
    if 'user_contact' not in st.session_state:
        st.session_state['user_contact'] = []
    # Append the feedback to the session state
    st.session_state.feedback.append(user_contact)

    # Here you would implement code to store feedback in S3
    # For demonstration, we're just printing the session state
    print(st.session_state['user_contact'])

###-----------------------------------------------------------
# Function to store feedback
###-----------------------------------------------------------
def store_feedback(feedback):
    # Check if 'feedback' already exists in session_state
    # If not, then initialize it as an empty list
    if 'feedback' not in st.session_state:
        st.session_state['feedback'] = []
    # Append the feedback to the session state
    st.session_state.feedback.append(feedback)

    # Here you would implement code to store feedback in S3
    # For demonstration, we're just printing the session state
    print(st.session_state['feedback'])


def main():
    #TODO: For some reason huggingface api token is not filled 
    #load_dotenv(override=True)

    #st.session_state.history = []

    # Streamlit app layout
    st.title("WeCloudData Chatbot System")
    st.write("Ask any question about WeCloudData and get an accurate and eloquent answer.")

    # Display the conversation history
    #for i, (q, a) in enumerate(st.session_state.history):
        #st.write(f"Q{i+1}: {q}")
        #st.write(f"A{i+1}: {a}")

    # Input from the user
    question = st.text_input("Enter your question:")

    if st.button("Get Answer... "):

        with st.spinner("Generating response..."):
            response_placeholder = st.empty()
            # Generate the response
            def generate_response(response_placeholder, question):
              msg = ""
              response = requests.get(url=backend_url, params={'query': question})
              response = response.json()["response"]
              response = response[response.index("{")+1:response_Ens.rindex("}")].strip()
              response = response_Ens.split("Answer": )[1]
              msg += response
              response_placeholder.write(msg)

              # Update the history after the response is generated
              # update_history(question, msg)

            #Use asyncio to run the async function
            generate_response(response_placeholder=response_placeholder, question=question)


    # Feedback section
    st.write("### Provide Feedback Rate")
    #feedback = st.selectbox("Please provide your feedback rate (from 1-very good to 5-very bad):", [1, 2, 3, 4, 5])

    feedback_options = {
    "1. Excellent": 1,
    "2. Good": 2,
    "3. Average": 3,
    "4. Below Average": 4,
    "5. Poor": 5
}

    feedback_text = st.selectbox(
        "Please provide your feedback rate:",
        list(feedback_options.keys())
    )

    feedback_value = feedback_options[feedback_text]

    if st.button("Submit Feedback"):
        #store_feedback("This is a sample feedback.")
        store_feedback(feedback_value)
        st.success("Thank you for your feedback!")


    # Functionality to review feedback (for demonstration purposes)
    #if st.button("Review Feedback"):
    #    st.write("Collected Feedback:")
    #    for fb in st.session_state.feedback:
    #        st.write(fb)

    st.write("### For more personalized help or follow-up")
    user_contact = st.text_input("Please provide your email address:")

    if st.button("Submit your Info"):
        store_userinfo(user_contact)
        st.success("Thank you for your info!")

###-----------------------------------------------------------

# Function to update the history
###-----------------------------------------------------------
def update_history(question, answer):
    st.session_state.history.append((question, answer))
    if len(st.session_state.history) > 0:
        st.session_state.history.pop(0)

if __name__ == "__main__":
  main()