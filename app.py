"""Streamlit app for conversational RAG with reddit posts."""

import streamlit as st
from Reddit import Reddit

# Initialize session state to store conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "posts_loaded" not in st.session_state:
    st.session_state.posts_loaded = False

st.set_page_config(page_title="Chat Reddit")
st.title("Chat Reddit")

# User input for category and number of posts
col1a, col2a = st.columns(2)
with col1a:
    category = st.selectbox("Category", options=["hot", "new", "top"])
with col2a:
    number_of_posts = st.number_input("Number of posts", 1, 10000)

subreddits = st.text_input(
    "Enter subreddits here. If multiple, separate with a comma (e.g. investing, wallstreetbets)"
)

# User input for subreddits
col1b, col2b, col3b = st.columns([1.1, 1.2, 8])
with col1b:
    run = st.button("Load")
with col2b:
    clear = st.button("Clear")

# Process and load Reddit posts if "Load" button is pressed
if run:
    with col3b:
        with st.status("Running") as status:
            st.session_state.processor = Reddit()
            loaded = st.session_state.processor.load_reddit(
                subreddits, category, number_of_posts
            )
            if loaded:
                st.session_state.posts_loaded = True  # Mark posts as loaded
                # Clear conversation history when new posts are loaded
                st.session_state.messages = []
                initiated_prompt = f"Hey! The {category} {number_of_posts} posts from {subreddits} have been loaded. Start your conversation below."
                st.session_state.clear_prompt = f"Messaged cleared. You are chatting with the {category} {number_of_posts} posts from {subreddits}."
                st.session_state.messages.append(
                    {"role": "assistant", "content": initiated_prompt}
                )
                status.update(label="Loaded", state="complete")
            else:
                status.update(
                    label=f"The subreddit '{subreddits}' is unavailable. Please check the spelling or try another subreddit.",
                    state="error",
                )

if clear:
    with col3b:
        with st.status("Clearing") as status:
            st.session_state.messages = []
            st.session_state.messages.append(
                {"role": "assistant", "content": st.session_state.clear_prompt}
            )
            status.update(label="Cleared", state="complete")

# Only display the conversation interface if posts have been loaded
if st.session_state.posts_loaded:
    # Display the conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            # TODO: Check if I can stream (st.write_stream)
            response = st.session_state.processor.query(prompt)
            st.markdown(response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
# TODO: Feedback button pointing somewhere.
# TODO: Could do a semantic similarity to find the common themes.
