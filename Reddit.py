"""A class to load reddit posts and allow a user to have a conversation with the posts"""

import uuid
import os
import prawcore
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders.reddit import RedditPostsLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain


class Reddit:
    """A class to load reddit posts and allow a user to have a conversation with the posts"""

    def __init__(self):
        """Inititate the reddit class"""
        self.__load_models()

    def __load_models(self):
        """Load the LLM and Embedding models"""
        self.llm = ChatGroq(
            model="llama3-8b-8192",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        inference_api_key = os.getenv("HUGGING_FACE_INFERENCE_API")
        self.embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=inference_api_key,
            model_name="sentence-transformers/all-MiniLM-l6-v2",
        )

    def load_reddit(self, subreddit, category, number_posts):
        """Load the posts from reddit"""
        loader = RedditPostsLoader(
            client_id=os.getenv("REDDIT_CI"),
            client_secret=os.getenv("REDDIT_CS"),
            user_agent="rag demonstration",
            categories=[category],  # List of categories to load posts from
            mode="subreddit",
            search_queries=subreddit.split(
                ", "
            ),  # List of subreddits to load posts from
            number_posts=number_posts,  # Default value is 10
        )
        try:  # TODO: prawcore.exceptions.Redirect
            documents = loader.load()
        except prawcore.exceptions.Redirect:
            return False
        self.vectorstore = FAISS.from_documents(documents, embedding=self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})
        return True

    def query(self, user_query):
        """Allow a user to query the posts and store the conversation history."""

        ### Contextualize question (including history) ###
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, contextualize_q_prompt
        )

        # Incorporate the retriever into a question-answering chain.
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Please note that the context has not been written "
            "by me but instead is a summarisation of other individuals blog posts"
            "\n\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        rag_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain
        )

        ### Statefully manage chat history ###
        store = {}

        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in store:
                store[session_id] = ChatMessageHistory()
            return store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        # TODO: Bring in more sophisticated rag techniques (includes move to langraph)
        # https://github.com/NVIDIA/GenerativeAIExamples/blob/main/RAG/notebooks/langchain/agentic_rag_with_nemo_retriever_nim.ipynb
        # TODO: Can do a loop to make sure the number of search kwargs doesn't go over the token limit.

        result = conversational_rag_chain.invoke(
            {"input": user_query}, config={"configurable": {"session_id": uuid.uuid4()}}
        )
        # TODO: Investigate streaming
        return result["answer"]
