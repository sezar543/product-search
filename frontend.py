
from QA_WCD_chatbot_Streamlit_2 import Setup_llm, qa_ENSEM_Run


import streamlit as st
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables.config import RunnableConfig


import asyncio


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

    if "runnable" not in st.session_state:
        model = Setup_llm()
        prompt = ChatPromptTemplate.from_messages(
            [
              (  "system",
                 """You are an intelligent and very knowledgeable chatbot operating for the site https://weclouddata.com/.
                    you are called "Weclouddata Chatbot".You answers in the conversation to maintain context and adjust your responses accordingly.
                    Provide detailed and accurate responses based on the documentation scope. 
                    Prevent hallucinations or providing unverified information. 
                    For out-of-scope questions, do not attempt to answer but inform the user that the question is beyond the current scope.

                    When you respond:
                    - Acknowledge the user's current question and any relevant context from previous interactions.
                    - Provide a comprehensive answer that covers all aspects of the question based on available documentation.
                    - Suggest additional information or resources on the topic when relevant.
                    - Direct the user to relevant links for more information.
                    - Ask for the user's email if more personalized help or follow-up is needed.
                    - Encourage further questions or clarifications to keep the conversation interactive.
                    - Format your response in JSON.

                    If a question is out of scope:
                    - Inform the user that the question is beyond the current scope of the chatbot,
                    - Suggest contacting support or visiting the documentation for more information.

                    
                    Current question: {{question}}
                    """
            ),
            ("human", "{question}"),
            ]
        )
        st.session_state.runnable = prompt | model | StrOutputParser()

        st.session_state.history = []

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
            async def generate_response():
                msg = ""
                async for response in st.session_state.runnable.astream(
                    {"question": question},
                    config=RunnableConfig(callbacks=[]),
                ):

                    response_Ens = qa_ENSEM_Run(question)
                    response=response_Ens.split("Answer:")[1]
                    msg += response
                    response_placeholder.write(msg)

                # Update the history after the response is generated
                # update_history(question, msg)

            #Use asyncio to run the async function
            asyncio.run(generate_response())


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