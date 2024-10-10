import os
import tempfile

from decouple import config
import streamlit as st

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

os.environ["OPENAI_API_KEY"] = config("OPENAI_API_KEY")
persist_directory = 'db'


def process_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name

    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()

    os.remove(temp_file_path)

    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=400,
    )
    chunks = text_spliter.split_documents(documents=docs)
    return chunks


def load_existing_vector_store():
    if os.path.exists(os.path.join(persist_directory)):
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=OpenAIEmbeddings()
        )
        return vector_store
    return None


def add_to_vector_store(chunks, vector_store=None):
    if vector_store:
        vector_store.add_documents(chunks)
    else:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=OpenAIEmbeddings(),
            persist_directory=persist_directory,
        )
        return vector_store


def ask_question(model, query, vector_store):
    llm = ChatOpenAI(model=model)
    retriver = vector_store.as_retriever()

    system_prompt = '''
    Use the provided context to answer questions.
    - If the answer is not found within the context,
    inform the user that the information is out of context.
    - Then, proceed to answer the question based on your general knowledge.

    **Response Format:**
    - Answer in Markdown with well-structured formatting.

    **Current Context:**
    {context}
    '''
    messages = [('system', system_prompt)]
    for message in st.session_state.messages:
        messages.append((message.get('role'), message.get('content')))
    messages.append(('human', '{input}'))

    prompt = ChatPromptTemplate.from_messages(messages)

    question_answer_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )
    chain = create_retrieval_chain(
        retriever=retriver,
        combine_docs_chain=question_answer_chain
    )
    response = chain.invoke({'input': query})
    return response.get('answer')


vector_store = load_existing_vector_store()

st.set_page_config(
    page_title='Personal chat',
    page_icon='🤖',
)
st.header('🤖 Chat with your documents')


with st.sidebar:
    st.header('Upload documents 📄')
    uploaded_files = st.file_uploader(
        label='Do upload of pdf documents',
        type=['pdf'],
        accept_multiple_files=True,
    )
    if uploaded_files:
        with st.spinner('processing documents...'):
            all_chunks = []
            for uploaded_file in uploaded_files:
                chunks = process_pdf(file=uploaded_file)
                all_chunks.extend(chunks)
            vector_store = add_to_vector_store(
                chunks=all_chunks,
                vector_store=vector_store,
            )

    model_options = [
        'gpt-3.5-turbo',
        'gpt-4',
        'gpt-4-turbo',
        'gpt-4o-mini',
        'gpt-4o',
    ]
    selected_model = st.sidebar.selectbox(
        label='Choose a LLM model',
        options=model_options
    )

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

question = st.chat_input('How can I help ?')

if vector_store and question:
    for message in st.session_state.messages:
        st.chat_message(message.get('role')).write(message.get('content'))

    st.chat_message('user').write(question)
    st.session_state.messages.append({'role': 'user', 'content': question})

    with st.spinner('looking for the answer...'):
        response = ask_question(
            model=selected_model,
            query=question,
            vector_store=vector_store,
        )
        st.chat_message('ai').write(response)
        st.session_state.messages.append({'role': 'ai', 'content': response})
