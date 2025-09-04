from typing import Set

from backend.core import run_llm
import streamlit as st

st.set_page_config(page_title="Language Documentation Bot", page_icon="💬", layout="centered")

# Sidebar: User information (profile picture, name, email)
with st.sidebar:
    st.header("User")

    # Defaults
    default_avatar_url = "https://avatars.githubusercontent.com/u/9919?s=200&v=4"
    default_user_name = "Jane Doe"
    default_user_email = "jane.doe@example.com"

    st.image(default_avatar_url, width=120)
    st.write(f"**{default_user_name}**")
    st.caption(default_user_email)

# WhatsApp-like styles
st.markdown(
    """
    <style>
      .chat-container {max-width: 860px; margin: 0 auto;}
      .header-bar {position: sticky; top: 0; z-index: 10; background: #f0f2f5; padding: 12px 14px; border-bottom: 1px solid #e0e0e0; border-top-left-radius: 8px; border-top-right-radius: 8px;}
      .header-title {font-weight: 600; font-size: 16px;}
      .bubble {position: relative; padding: 10px 12px; border-radius: 7.5px; margin: 6px 0; max-width: 80%; line-height: 1.3;}
      .me {background: #d9fdd3; color: #111; margin-left: auto; border-bottom-right-radius: 2px;}
      .you {background: #ffffff; color: #111; margin-right: auto; border-bottom-left-radius: 2px; border: 1px solid #ececec;}
      .timestamp {display:block; text-align: right; font-size: 11px; opacity: .6; margin-top: 4px;}
      .source-list {margin-top: 6px; font-size: 12px; color: #1778f2;}
      .source-list a {color: #1778f2; text-decoration: none;}
      .source-list a:hover {text-decoration: underline;}
      .chat-scroll {height: calc(100vh - 240px); overflow-y: auto; padding: 8px 8px 0 8px; background: #efeae2; border-bottom-left-radius: 8px; border-bottom-right-radius: 8px;}
      /* Hide default streamlit elements that clash with custom layout */
      .stChatMessage {display: none}
    </style>
    """,
    unsafe_allow_html=True,
)

# Session state
if ("user_prompt_history" not in st.session_state
        and "chat_answers_history" not in st.session_state
        and "chat_history" not in st.session_state
        and "assistant_sources" not in st.session_state):
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_answers_history"] = []
    st.session_state["chat_history"] = []  # list[tuple[str, str]] of (role, content)
    st.session_state["assistant_sources"] = []  # list[list[str]] aligned with answers


def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i + 1}. {source}\n"
    return sources_string


# Input
prompt = st.chat_input("Type a message")

if prompt:
    with st.spinner("Generating response..."):
        generated_response = run_llm(query=prompt, chat_history=st.session_state["chat_history"])
        sources = set(
            [doc.metadata.get("source", "") for doc in generated_response.get("source_documents", []) if doc and doc.metadata.get("source")]
        )

        formatted_response = (
            f"{generated_response['result']} \n\n {create_sources_string(sources)}"
        )

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)
        st.session_state["assistant_sources"].append(list(sources))
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", generated_response["result"]))


# Render messages in WhatsApp-like bubbles
with st.container():
    # Build one HTML block so structure remains intact
    bubbles = []

    if st.session_state["chat_answers_history"]:
        for answer, user_query, sources in zip(
            st.session_state["chat_answers_history"],
            st.session_state["user_prompt_history"],
            st.session_state["assistant_sources"],
        ):
            bubbles.append(f'<div class="bubble me">{user_query}</div>')

            safe_answer = answer.replace("\n", "<br>")
            sources_html = ""
            if sources:
                links = []
                for i, s in enumerate(sorted(set(sources))):
                    label = f"Source {i+1}"
                    links.append(f"<a href=\"{s}\" target=\"_blank\">{label}</a>")
                sources_html = f"<div class=\"source-list\">{' • '.join(links)}</div>"

            bubbles.append(f'<div class="bubble you">{safe_answer}{sources_html}</div>')

    html = (
        '<div class="chat-container">'
        '<div class="header-bar"><span class="header-title">💬 Language Documentation Bot</span></div>'
        '<div class="chat-scroll">' + "".join(bubbles) + '</div>'
        '</div>'
    )

    st.markdown(html, unsafe_allow_html=True)
