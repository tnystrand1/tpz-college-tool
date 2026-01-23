import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd

st.set_page_config(page_title="The Possible Zone | College Search", page_icon="🦉", layout="centered")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Work+Sans:wght@300;400;600;800&display=swap');
        html, body, [class*="css"] { font-family: "Work Sans", sans-serif; color: #101921; }
        h1, h2, h3 { font-family: 'Work Sans', sans-serif; font-weight: 800; color: #101921; }
        .stChatInput { bottom: 20px; }
        .footer { position: fixed; left: 0; bottom: 0; width: 100%; background-color: #101921; color: white; text-align: center; padding: 10px; font-size: 12px; z-index: 1000; }
    </style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hoot! 🦉 Welcome to **The Possible Zone College Search**!\n\nI am here to help you find your perfect school. Tell me a little about yourself!\n\n*What do you want to study, or do you have a dream city in mind?*", "avatar": "🦉"}
    ]

with st.sidebar:
    st.header("⚙️ Settings")
    if "OPENROUTER_API_KEY" in st.secrets:
        api_key = st.secrets["OPENROUTER_API_KEY"]
    else:
        api_key = st.text_input("OpenRouter API Key", type="password")

    if "GCP_PROJECT_ID" in st.secrets:
        project_id = st.secrets["GCP_PROJECT_ID"]
    else:
        project_id = st.text_input("GCP Project ID", value="tpzcollegesearch")

    st.markdown('---')
    st.subheader("🦉 How to use")
    st.markdown("**1. Introduce yourself!**")
    st.caption("Tell me your name, where you live, or what grade you are in.")
    
    st.markdown("**2. Share your goals**")
    st.caption("What do you want to study? Do you have a dream city?")
    
    st.markdown("**3. Get specific**")
    st.caption("Mention your SAT score or budget to find the best matches.")
    
    st.markdown('---')
    # --- RESET BUTTON ---
    if st.button("🔄 Start Over", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hoot! 🦉 Ready for a fresh start. What are we looking for now?", "avatar": "🦉"}
        ]
        st.rerun()

st.title("THE **POSSIBLE** ZONE")
st.markdown("<h3 style='color: #00b2b1;'>AI College Search Engine</h3>", unsafe_allow_html=True)

if not api_key:
    st.info("👋 Welcome! Please enter your **OpenRouter Key** in the sidebar to start.")
    st.stop()

@st.cache_resource
def get_tools(api_key, project_id):
    if "gcp_service_account" in st.secrets:
        creds = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"]
        )
        client = bigquery.Client(credentials=creds, project=project_id)
    else:
        client = bigquery.Client(project=project_id)

    llm = ChatOpenAI(
        model="anthropic/claude-3.5-sonnet",
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0
    )
    return client, llm

try:
    client, llm = get_tools(api_key, project_id)
except Exception as e:
    st.error(f"Connection Error: {e}")
    st.stop()

# --- 1. THE ROUTER PROMPT ---
router_prompt = PromptTemplate.from_template("""
    You are a friendly College Mentor named 'Owl'.
    Your goal is to decide if you need to SEARCH for schools or just CHAT with the student.

    CHAT HISTORY:
    {history}

    LATEST USER MESSAGE:
    {input}

    INSTRUCTIONS:
    1. **If the request is VAGUE (Only 1 detail)**: DO NOT SEARCH.
       - Example: 'I want to study Physics' -> CHAT: 'Physics is great! Do you have a specific city or state in mind?'
       - Example: 'Schools in Boston' -> CHAT: 'Boston has many schools! What do you want to study?'
    2. **If the request is SPECIFIC (2+ details)**: SEARCH.
       - Example: 'Physics in Boston' -> SEARCH: Physics schools in Boston
       - Example: 'Cheap schools for nursing' -> SEARCH: Affordable nursing schools
    3. **If the user explicitly asks to search**: SEARCH.
       - Example: 'Just show me the list' -> SEARCH: ...

    YOUR RESPONSE (Start with CHAT: or SEARCH:):
""")

# --- 2. THE SQL PROMPT ---
sql_prompt = PromptTemplate.from_template("""
    You are a BigQuery SQL expert. Write a valid StandardSQL query.
    Dataset: `{project}.most_recent_cohorts_institution.collegedata`

    IMPORTANT: All columns are STRING. You MUST use SAFE_CAST() for any number comparison.

    SCHEMA:
    - INSTNM (Name)
    - CITY, STABBR (Location)
    - ADM_RATE (STRING -> Cast to FLOAT64)
    - SAT_AVG (STRING -> Cast to INT64)
    - C150_4 (Graduation Rate, STRING -> Cast to FLOAT64)
    - TUITIONFEE_IN (In-State Cost, STRING -> Cast to FLOAT64)
    - TUITIONFEE_OUT (Out-State Cost, STRING -> Cast to FLOAT64)
    - UGDS (Size, STRING -> Cast to INT64)
    - UGDS_WHITE, UGDS_BLACK, UGDS_HISP, UGDS_ASIAN (Diversity, Cast to FLOAT64)
    - CONTROL (1=Public, 2=Private), ICLEVEL (1=4yr, 2=2yr), MD_EARN_WNE_P10 (Earnings)
    - PCIP11 (CS), PCIP52 (Business), PCIP51 (Health) -> Decimals (0.15 = 15%)

    STRATEGY GUIDE:
    1. **'Boston'**: Use CITY IN ('Boston', 'Cambridge', 'Chestnut Hill', 'Medford', 'Waltham', 'Newton', 'Brookline', 'Quincy', 'Somerville')
    2. **Majors**: WHERE SAFE_CAST(PCIP52 AS FLOAT64) > 0.05
    3. **SAT**: SAFE_CAST(SAT_AVG AS INT64) BETWEEN ([Score] - 150) AND ([Score] + 150)
    4. **Community Colleges**: WHERE SAFE_CAST(ICLEVEL AS INT64) = 2. DO NOT filter by SAT.

    CRITICAL RULES:
    1. Use SAFE_CAST(Column AS TYPE) for numbers.
    2. Filter NULLs ONLY for the metric being requested.
    3. Return ONLY SQL.

    Question: {question}
""")

answer_prompt = PromptTemplate.from_template("""
    Question: {question}
    SQL Query: {query}
    Data Results: {data}
    Summarize the data clearly for a high school student.
""")

# --- CHAT UI LOGIC ---

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message.get("avatar")):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": prompt, "avatar": "👤"})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🦉"):
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-4:]])
        
        router_chain = router_prompt | llm | StrOutputParser()
        decision = router_chain.invoke({"history": history_text, "input": prompt})
        
        if decision.startswith("CHAT:"):
            response_text = decision.replace("CHAT:", "").strip()
            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text, "avatar": "🦉"})
        
        elif decision.startswith("SEARCH:"):
            search_query = decision.replace("SEARCH:", "").strip()
            
            status_container = st.status("🦉 Looking for matches...", expanded=True)
            try:
                status_container.write(f"🔍 Searching for: {search_query}")
                
                chain_1 = sql_prompt | llm | StrOutputParser()
                sql_query = chain_1.invoke({"question": search_query, "project": project_id})
                clean_sql = sql_query.replace("```sql", "").replace("```", "").strip()
                
                status_container.write("👀 Reading the results...")
                query_job = client.query(clean_sql)
                results = query_job.result()
                rows = [dict(row) for row in results]
                
                if not rows:
                    status_container.update(label="❌ No matches found", state="error", expanded=True)
                    st.warning("I couldn't find any schools that match exactly. Try widening your search?")
                    st.session_state.messages.append({"role": "assistant", "content": "I couldn't find any schools that match exactly.", "avatar": "🦉"})
                else:
                    status_container.write("✨ Writing up your answer...")
                    chain_2 = answer_prompt | llm | StrOutputParser()
                    answer = chain_2.invoke({"question": search_query, "query": clean_sql, "data": str(rows)})
                    
                    status_container.update(label="✅ Found them!", state="complete", expanded=False)
                    
                    st.markdown(answer)
                    st.dataframe(pd.DataFrame(rows))
                    st.session_state.messages.append({"role": "assistant", "content": answer, "avatar": "🦉"})

            except Exception as e:
                status_container.update(label="⚠️ Oops, something went wrong", state="error")
                st.error(f"Error details: {e}")

st.markdown('<div class="footer">The Possible Zone © 2026 | Internal Data Tool</div>', unsafe_allow_html=True)