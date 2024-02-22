"""
##### IMPORTANT NOTES #####
1. Edit setup-environment.sh as you may have to remove the "3" in python3 and pip3 depending on your system
2. Run "chmod +x setup-environment.sh" in your terminal
3. Run "source ./setup-environment.sh" in your terminal
4. Authenticate with AWS and then run "streamlit run [PYTHON-APP-FILE-NAME].py" in your terminal.  A browser window/tab will appear with the application.
#####
"""

import boto3
import io
import json
import pandas
import streamlit as st

from xml.etree import ElementTree as ET

# Set Streamlit page configuration
st.set_page_config(page_title="Amazon Bedrock Query Intent", layout="wide")
st.title("🤖 Amazon Bedrock Query Intent")

st.markdown(
    """<style>
.stMarkdown ul li {
    line-height: .3rem !important;
    font-size: .8rem !important;
}</style>""",
    unsafe_allow_html=True,
)


existing_queries = []


def call_llm(prompt):
    session = boto3.Session()
    bedrock_runtime = session.client("bedrock-runtime")
    claude_bedrock_model_id = "anthropic.claude-instant-v1"
    claude_inference_configuration = {
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "max_tokens_to_sample": MAX_TOKENS,
        "stop_sequences": ["\n\nHuman:"],
    }

    claude_inference_configuration["prompt"] = prompt
    accept = "application/json"
    contentType = "application/json"
    body = json.dumps(claude_inference_configuration)

    llm_full_response = bedrock_runtime.invoke_model(
        body=body,
        modelId=claude_bedrock_model_id,
        accept=accept,
        contentType=contentType,
    )
    llm_response_completion_only = json.loads(llm_full_response.get("body").read()).get(
        "completion"
    )

    return llm_response_completion_only


def determine_single_query_intent(query):
    with output_container:
        intent = determine_query_intent(query)
        st.write(f"Original Question: {intent['question']}")
        st.code(intent["completion"], language="xml", line_numbers=False)


def determine_query_intent(query):
    determine_intent_prompt = """
    <role>
    1. You are a discerning AI that can determine what a user's search intent given a user query.
    </role>

    <user_personas>
    1. Clinical Staff
    2. Medical Doctors
    3. Doctors doing Continuing Education
    4. Researchers
    </user_personas>

    <task_instructions>
    1. Take a deep breath and focus on medical and clinical information.
    2. You will determine the user's search intent and categorize it into ONLY ONE of the categories listed in the <intent_categories></intent_categories> xml tags.
    3. You will explain your thought process, wrap it into <thought_process></thought_process> xml tags.
    4. Based on the tools defined for each category, select a tool to use and wrap your response in <tool></tool> xml tags.
    5. You will format your answer based on the XML style format provided in the <answer_format></answer_format> xml tags.
    </task_instructions>

    <task_guidance>
    1. The types and roles of your users are within the <user_personas></user_personas> xml tags above.
    2. You MUST ONLY give your categorization and thought process, NOTHING ELSE.
    3. You MUST BE HONEST if you don't know the answer.
    4. You MUST NOT hallucinate.
    5. You MUST NOT include the prompt you were given.
    6. You MUST NOT directly answer the user's question without using a tool.
    7. You MUST ONLY give your categorization and thought process, NOTHING ELSE.
    </task_guidance>

    <intent_categories>
    * Clinical Decision Support
    ** Clinical Decision Support Category Description: The user needs to review information to support and guide a clinical decision.
    ** Clinical Decision Support Category Tool:  This tool retrieves information from the Clinical Decision Support Knowledge Base.

    * Clinical Research
    ** Clinical Research Category Description: The user needs information or education related to the medical and clinical domains.
    ** Clinical Research Category Tool:  This tool retrieves information from the Clinical Research Knowledge Base.
    </intent_categories>

    <answer_format>
    <response>
    <query>replace with the user's query here</query>
    <category>replace with user's query intent category here</category>
    <tool>replace with the right tool to use based on the user's query intent category here</tool>
    <thought_process>replace with your thought process here</thought_process>
    </response>
    </answer_format>

    <user_query>
    {user_query}
    </user_query>
        """
    determine_intent_prompt = determine_intent_prompt.replace("{user_query}", query)

    llm_claude_prompt_template = """
    \n\nHuman: {instructions}
    \n\nAssistant:
    """

    final_prompt = llm_claude_prompt_template.replace(
        "{instructions}", determine_intent_prompt
    )

    llm_output = call_llm(prompt=final_prompt)

    structured_response = {}
    structured_response["question"] = query
    structured_response["completion"] = llm_output

    return structured_response


# Sidebar info
with st.sidebar:
    st.markdown("## Inference Parameters")
    TEMPERATURE = st.slider(
        "Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.1
    )
    TOP_P = st.slider("Top-P", min_value=0.0, max_value=1.0, value=0.9, step=0.01)
    TOP_K = st.slider("Top-K", min_value=1, max_value=500, value=10, step=5)
    MAX_TOKENS = st.slider("Max Token", min_value=0, max_value=2048, value=1024, step=8)

model_kwargs = {
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "top_k": TOP_K,
    "max_tokens_to_sample": MAX_TOKENS,
}

if (
    "app_mode" not in st.session_state
    or st.session_state.app_mode == None
    or st.session_state.app_mode == ""
):
    st.session_state.app_mode = "existing_queries"


tab1, tab2 = st.tabs(["Provided Samples", "Try your own!"])
main_container = st.container()
with main_container:
    with tab1:
        st.write("Provided Sample Queries ")
        existing_queries_container = st.container()
        with existing_queries_container:
            existing_queries = [
                "research on the effectiveness of levaquin",
                "diagnositc criteria for lupus",
                "Chrohns patient presents with lactose intolerance, how do I decide if it's related to their Chrohns?",
                "Tell me about osteoporosis.",
                "Explain osteoporosis to me.",
                "I need to make a decision on treatment plans for sacroilitis",
                "Treatment options for cervical disc degeneration.",
                "leukemia 6mp adolescent",
                "information on the comorbidity of chrohns and RA",
            ]

        for existing_query in existing_queries:
            st.markdown("- " + existing_query)
        st.divider()
        existing_queries_results_container = st.container()

    with tab2:
        input_container = st.container()
        output_container = st.container()
        with output_container:
            query_try = st.text_input(
                label="Enter your query",
                key="query_try",
            )
            query_submit = st.button(label="Submit Query", key="query_submit")

            if query_submit:
                st.session_state.app_mode = "single_query"
                st.divider()
                st.write("Intent Determination Results")

                determine_single_query_intent(query_try)


def run_app():

    with tab1:
        queries_submit = st.button(label="Submit Queries", key="queries_submit")
        if queries_submit:
            st.session_state.app_mode = "existing_queries"
            with existing_queries_container:
                my_bar = st.progress(text="Determing query intent...", value=0)

            # with st.spinner(text="Determining query intent from provided samples."):
            response_xml = "<examples>"
            i = 0
            list_count = 0
            if st.session_state.app_mode == "existing_queries":
                existing_query_list_length = len(existing_queries)
                for query in existing_queries:
                    llm_output = determine_query_intent(query)
                    response_xml = response_xml + llm_output["completion"]
                    print("rounding...")
                    i += round(((100 / existing_query_list_length)), 1)
                    y = round(i / 100, 1)
                    print(y)

                    list_count += 1
                    my_bar.progress(y)
                    # f"Query #{list_count} out of {existing_query_list_length} processed."

                    print(f'Processing query: "{query}"')
                    st.session_state.update_text = "Complete"
                response_xml += "</examples>"
                response_xml = response_xml.replace("\n", "")
                display_results_data_frame(response_xml)


def color_rows(row):
    # https://www.w3.org/wiki/CSS/Properties/color/keywords
    color = (
        "background-color: teal; color: white;"
        if row["category"] == "Clinical Research"
        else "background-color: maroon; color: white;"
    )
    return [color] * len(row)


def display_results_data_frame(response_xml):
    try:
        x = ET.fromstring(response_xml)
        print("xml is valid")
    except Exception as e:
        print(e)

    results_df = pandas.read_xml(
        path_or_buffer=io.StringIO(response_xml),
        # xpath = 'examples',
        elems_only=True,
    )

    with existing_queries_results_container:
        st.write("Query Intent Determiniations...")
        st.dataframe(results_df.style.apply(color_rows, axis=1))

    with existing_queries_results_container:
        st.divider()
        st.write("Thought Process Log")
        for index, row in results_df.iterrows():
            # Print value of 'name' column
            st.markdown(
                f'<span style="color: yellow">Query:</span> {row["query"]}',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<span style="color: yellow">Determination:</span> {row["category"]}',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<span style="color: yellow">Decision Thought Process:</span>',
                unsafe_allow_html=True,
            )
            st.write(row["thought_process"], unsafe_allow_html=True)

            st.divider()

    return


run_app()
