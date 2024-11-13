
import os
from dotenv import find_dotenv, load_dotenv
import requests
import re

import streamlit as st
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables.config import RunnableConfig
import json
# import asyncio

load_dotenv(dotenv_path="../.env")
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

def store_userinfo(user_contact):
    # Check if 'feedback' already exists in session_state
    # If not, then initialize it as an empty list
    if 'user_contact' not in st.session_state:
        st.session_state['user_contact'] = []
    # Append the feedback to the session state
    st.session_state.feedback.append(user_contact)

    # Here you would implement code to store feedback in S3
    # For demonstration, we're just printing the session state
    # print(st.session_state['user_contact'])

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
    # print(st.session_state['feedback'])

def extract_response(response_data):
    """
    Extract and return only the response part from the response_data dictionary.
    """   
    return response_data.get('response', '')

def format_response(response_data):
    """
    Convert the string response_data to a dictionary and format each value to print on a new line.
    """
    if isinstance(response_data, str):
        try:
            # Attempt to parse as JSON
            response_data = json.loads(response_data)
        except json.JSONDecodeError:
            # Parsing failed, so response_data remains a str
            pass

    # If response_data is now a dict, check for "Answer" key
    if isinstance(response_data, dict):
        if "Answer" not in response_data:
            return response_data['response']

    # If it's still a str, try to find "Answer" text manually
    elif isinstance(response_data, str):
        if str('Answer') not in response_data:
            return response_data
        else:
            return "I'm sorry, but we cannot provide you with an answer at this time for that question. Please feel free to ask another question."

    formatted_text = ""
    
    # Access and format each key's value
    for key in ["Acknowledgement", "Answer", "References", "Additional_info", "Contact_info", "User_info", "Feedback_request", "Thank_you"]:
        if key in response_data:
            value = response_data[key]
            if key == "References" and isinstance(value, dict):
                # Use <p> tags for references for better control over line spacing
                references = "<p>" + "</p><p>".join([f"{k}: {v}" for k, v in value.items()]) + "</p>"
                formatted_text += references
            elif isinstance(value, str):
                # Preprocess the string to replace \n with a space
                formatted_value = value.replace("\n", " ")
                # Wrap the value in <p> tags
                formatted_text += f"<p>{formatted_value}</p>"
            else:
                formatted_text += f"<p>{value}</p>"
    return formatted_text.strip()  # Remove any extra whitespace from the end

def generate_response(response_placeholder, question):
    msg = ""
    print("Generating response...")

    # Send the request to FastAPI
    response = requests.get(url=f"{BACKEND_URL}/chatbot", params={'query': question})
    
    # Parse the response JSON
    response_json = response.json()

    # Check if response contains valid JSON with a structure
    if isinstance(response_json, dict):
        # Format the response JSON for display
        ext_msg = extract_response(response_json)
        msg = format_response(ext_msg)
    else:
        # If response isn't structured, use plain text
        msg = response_json if isinstance(response_json, str) else ""

    # Display the formatted message
    response_placeholder.markdown(msg, unsafe_allow_html=True)

# Adding a section for the TEST_PATH endpoint
st.header("Reverse Text Endpoint (/test)")

user_input = st.text_input("Enter some text:")
if st.button("Submit"):

    if user_input:
        # Prepare payload to send to the API Gateway
        payload = {
            'input': user_input  # Pass user input to the Lambda function
        }

        headers = {
            'Content-Type': 'application/json'  # Ensure the content type is set correctly
        }

        try:
            # Send POST request to the API Gateway /test endpoint
            response = requests.post(f"{BACKEND_URL}/test", json=payload, headers=headers)

            # Check if the response status code indicates success (200)
            if response.status_code == 200:
                try:
                    # Parse the JSON response content
                    result = response.json()
                    st.write("Response from API Gateway:")
                    st.json(result)
                except json.JSONDecodeError:
                    st.error("Error: Received response is not valid JSON")
                    st.write(response.text)  # Show raw text if JSON parsing fails
            else:
                st.error(f"Error: Received status code {response.status_code}")
                st.write(response.text)  # Show error response content

        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")
    else:
        st.write("Please enter some text.")

def main():
    # Set up Streamlit layout
    st.title("WeCloudData Chatbot System")
    st.write("Ask any question about WeCloudData and get an accurate and eloquent answer.")

    # Input from the user
    question = st.text_input("Enter your question:")
    if st.button("Get Answer..."):
        with st.spinner("Generating response..."):
            response_placeholder = st.empty()
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