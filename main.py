# Evaluate a RAG App

import os
import streamlit as st
from enum import Enum
from langchain_groq import ChatGroq
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain.evaluation.qa import QAEvalChain
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter

CREATIVITY=0
# os.environ["TOKENIZERS_PARALLELISM"] = False


class ModelType(Enum):
    GROQ='GroqCloud'
    OPENAI='OpenAI'


class LLMModel:
    def __init__(self, model_provider: str) -> None:
        self.model_provider = model_provider

    def load(self, api_key=str):
        try:
            if self.model_provider==ModelType.GROQ.value:
                llm = ChatGroq(temperature=CREATIVITY, model="llama3-70b-8192", api_key=api_key) # model="mixtral-8x7b-32768"
            if self.model_provider==ModelType.OPENAI.value:
                llm = OpenAI(temperature=CREATIVITY, api_key=api_key)
            return llm
        
        except Exception as e:
            raise e


class LLMStreamlitUI:
    def __init__(self) -> None:
        pass

    def validate_api_key(self, key:str):
        if not key:
            st.sidebar.warning("Please enter your API Key")
            # st.stop()
        else:    
            if (key.startswith("sk-") or key.startswith("gsk_")):
                st.sidebar.success("Received valid API Key!")
            else:
                st.sidebar.error("Invalid API Key!")

    def get_api_key(self):
        
        # Get the API Key to query the model
        input_text = st.sidebar.text_input(
            label="Your API Key",
            placeholder="Ex: sk-2twmA8tfCb8un4...",
            key="api_key_input",
            type="password"
        )

        # Validate the API key
        self.validate_api_key(input_text)
        return input_text
    
    def evaluate_app(self, llm, real_qa, predictions):
        # Create an eval chain
        eval_chain = QAEvalChain.from_llm(
            llm=llm
        )

        # Have it grade itself
        graded_outputs = eval_chain.evaluate(
            real_qa,
            predictions,
            question_key="question",
            prediction_key="result",
            answer_key="answer"
        )

        response = {
            "predictions": predictions,
            "graded_outputs": graded_outputs
        }
        return response

    def create(self):
        try:
            # Set the page title for blog post
            st.set_page_config(page_title="Evaluate a RAG App")
            st.markdown("<h1 style='text-align: center;'>Evaluate a RAG App</h1>", unsafe_allow_html=True)

            # Select the model provider
            option_model_provider = st.sidebar.selectbox(
                    'Select the model provider',
                    ('GroqCloud', 'OpenAI')
                )

            # Input API Key for model to query
            api_key = self.get_api_key()

            with st.expander("Evaluate the quality of a RAG App"):
                st.write("""
                    To evaluate the quality of a RAG app, we will
                    ask it questions for which we already know the
                    real answers. That way we can see if the app is producing
                    the right answers or if it is hallucinating.
                """)
            
            # Upload a text file
            uploaded_file = st.file_uploader("Upload a text file", type="txt")
            
            query_text = st.text_input(
                "Enter a question you have already fact-checked:",
                placeholder="Write your question here",
                disabled= not uploaded_file
            )
            response_text = st.text_input(
                "Enter the real answer to the question:",
                placeholder="Write the confirmed answer here",
                disabled=not uploaded_file
            )

            submitted = st.button("Submit", disabled=not (uploaded_file and query_text and response_text))
            if submitted:
                if not api_key:
                    st.warning("Please insert your API Key", icon="⚠️")
                    st.stop()

                with st.spinner("Wait, please. I'm working on it..."):        
                    # Format file
                    documents = [uploaded_file.read().decode()]

                    text_splitter = CharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=0
                    )
                    texts = text_splitter.create_documents(documents)

                    # Create embeddings and store into FAISS vector db
                    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                    vectordb = FAISS.from_documents(texts, embeddings)

                    # Load the LLM model
                    llm_model = LLMModel(model_provider=option_model_provider)
                    llm = llm_model.load(api_key=api_key)

                    # Create retriever and QA chain
                    retriever = vectordb.as_retriever()
                    qachain = RetrievalQA.from_chain_type(
                         llm=llm,
                         chain_type="stuff",
                         retriever=retriever,
                         input_key="question"
                    )

                    # Create real QA dictionary
                    real_qa = [
                        {
                            "question": query_text,
                            "answer": response_text
                        }
                    ]
                    predictions=qachain.apply(real_qa)
                    print("Predictions:: ", predictions)

                    # Evaluate the app
                    response = self.evaluate_app(llm, real_qa, predictions)
                    print("Response:: ", response)
                    if response:
                        st.write("Question")
                        st.info(response["predictions"][0]["question"])

                        st.write("Real answer")
                        st.info(response["predictions"][0]["answer"])

                        st.write("Answer provided by the AI App")
                        st.info(response["predictions"][0]["result"])

                        st.write("Therefore, the AI App answer was")
                        st.info(response["graded_outputs"][0]["results"])

                        del api_key

        except Exception as e:
            st.error(str(e), icon=":material/error:")


def main():
    # Create the streamlit UI
    st_ui = LLMStreamlitUI()
    st_ui.create()


if __name__ == "__main__":
    main()