import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from langsmith import Client
from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from get_prompt import load_prompt, load_prompt_with_questions

#from dotenv import load_dotenv

# Load environment variables from .env
#load_dotenv()


st.set_page_config(page_title="GENERAL MEDICINE : Getting Started Class", page_icon="")
st.title("GENERAL MEDICINE")
button_css = """.stButton>button {
    color: #4F8BF9;
    border-radius: 50%;
    height: 2em;
    width: 2em;
    font-size: 4px;
}"""
st.markdown(f'<style>{button_css}</style>', unsafe_allow_html=True)

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

# Lesson selection dictionary
lesson_guides = {
    "Lesson 1: Basics of ADH Guide": {
        "file": "lc_guides/Basics_of_ADH_guide.txt",
        "description": "This lesson covers the basics of Basics of ADH Guide."
    },
    "Lesson 2: Growth Hormone Guide": {
        "file": "lc_guides/Growth_Hormone_guide.txt",
        "description": "This lesson focuses on Growth hormone guide."
    },
}

# Initialize LangSmith client
client = Client()

import os
openai_api_key = os.getenv("OPENAI_API_KEY")


# Lesson selection sidebar
lesson_selection = st.sidebar.selectbox("Select Lesson", list(lesson_guides.keys()))

# Display lesson content and description based on selection
lesson_info = lesson_guides[lesson_selection]
lesson_content = open(lesson_info["file"], "r").read()
lesson_description = lesson_info["description"]

# Radio buttons for lesson type selection
lesson_type = st.sidebar.radio("Select Lesson Type", ["Instructions based lesson", "Interactive lesson with questions"])

# Clear chat session if dropdown option or radio button changes
if st.session_state.get("current_lesson") != lesson_selection or st.session_state.get("current_lesson_type") != lesson_type:
    st.session_state["current_lesson"] = lesson_selection
    st.session_state["current_lesson_type"] = lesson_type
    st.session_state["messages"] = [AIMessage(content="Welcome! This short course will help you get started with General Medicine. Let me know when you're all set to jump in!")]

# Display lesson name and description
st.markdown(f"**{lesson_selection}**")
st.write(lesson_description)

# Message handling and interaction
def send_feedback(run_id, score):
    client.create_feedback(run_id, "user_score", score=score)

for msg in st.session_state["messages"]:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    else:
        st.chat_message("assistant").write(msg.content)

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        model = ChatOpenAI(streaming=True, callbacks=[stream_handler], model="gpt-4",openai_api_key=openai_api_key)

        if lesson_type == "Instructions based lesson":
            prompt_template = load_prompt(content=lesson_content)
        else:
            prompt_template = load_prompt_with_questions(content=lesson_content)

        chain = LLMChain(prompt=prompt_template, llm=model)

        response = chain(
            {"input": prompt, "chat_history": st.session_state.messages[-20:]},
            include_run_info=True,
            tags=[lesson_selection, lesson_type]
        )
        st.session_state.messages.append(HumanMessage(content=prompt))
        st.session_state.messages.append(AIMessage(content=response[chain.output_key]))
        run_id = response["__run"].run_id

        col_blank, col_text, col1, col2 = st.columns([1, 2, 1, 1])
        
        # with col_option:
        #     st.button('Explain this topic in brief')

        with col_text:
            st.text("Feedback:")

        with col1:
            st.button("👍", on_click=send_feedback, args=(run_id, 1))

        with col2:
            st.button("👎", on_click=send_feedback, args=(run_id, 0))

