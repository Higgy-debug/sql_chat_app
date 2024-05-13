import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.llms import LlamaCpp

# Database URIs
db_uris = {
    "All-Brands View": "mysql+mysqlconnector://cron:hirthickkesh@192.168.0.215:3306/brittania",
    "Internal View": "YOUR_INTERNAL_VIEW_URI_HERE",
    "Format View": "mysql+mysqlconnector://cron:hirthickkesh@192.168.0.215:3306/format",
    "Penetration View": "YOUR_INTERNAL_VIEW_URI_HERE",
    "PricePoint View": "mysql+mysqlconnector://cron:hirthickkesh@192.168.0.215:3306/PPU",
    "Flavour View": "mysql+mysqlconnector://cron:hirthickkesh@192.168.0.215:3306/Flavour",
}

# Connect to the database
def connect_database(db_name: str) -> SQLDatabase:
    db_uri = db_uris.get(db_name)
    if db_uri:
        return SQLDatabase.from_uri(db_uri)
    else:
        raise ValueError(f"Database '{db_name}' not found.")

# Load LLM model
@st.cache(allow_output_mutation=True)
def load_llm():
    return LlamaCpp(
        streaming=True,
        n_gpu_layers=-1,
        model_path='/home/hirthick/poc/llm/sqlcoder-7b-Mistral-7B-Instruct-v0.2-slerp.Q8_0.gguf',
        temperature=0.1,
        top_p=1,
        verbose=True,
        n_ctx=4096
    )

# Define SQL query chain
def get_sql_chain(db, llm):
    # Template for SQL prompt
    prompt_template = """
        You are a senior data analyst working with a dataset containing information about brand sales, market shares, and performance metrics.
        The data is organized into a table with the following columns: <SCHEMA>{schema}</SCHEMA>
        
        The dataset is used to track sales performance and market share of various brands and products across different regions and time periods.
      
        To calculate certain performance metrics, you should use the following KPI formulas:

        MS (Market Share) = Aggregate Sum of (Numerator/Denominator) 
        ND (Numeric Distribution) = Aggregate Sum of (Numerator/Denominator)
        Nielsen Value(Total Sales) = Sum(Numerator) / power(10, 10)
        Nielsen Volume = Sum(Numerator) / power(10, 11)
        PDO = Aggregate Sum of (Numerator/Denominator)
        Num of Dealers = Sum(Numerator)
        
        When calculating any of the above KPIs, consider only the rows where the 'Market' column has the value 'U+R' and the 'SL' column has the value 'SLY' .

        Based on the table schema, the provided KPI formulas, and the conversation history below, write a SQL query that answers the given question.
        
        For Example:
        Question: Calculate MS for Britannia for Cookies and ALL India in Dec-23?
        SQL Query:
        SELECT SUM(Numerator/Denominator) AS MS 
        FROM company_region
        WHERE Market = 'U+R'
            AND SL = 'SLY'
            AND KPI = 'MS'
            AND brit_top_50_companies LIKE 'BRITANNIA INDS'
            AND period = 'Dec-23'
            AND brit_seg3_670 = 'COOKIES'
            AND region = 'ALL INDIA';
        
        Question: Calculate MS for Britannia and ITC for Cookies and ALL India in Dec-23?
        SQL Query: 
        SELECT brit_top_50_companies AS Company, SUM(Numerator/Denominator) AS MS 
        FROM company_region
        WHERE Market = 'U+R'
            AND SL = 'SLY'
            AND KPI = 'MS'
            AND brit_top_50_companies IN ('BRITANNIA INDS', 'I T C')
            AND period = 'Dec-23'
            AND brit_seg3_670 = 'COOKIES'
            AND region = 'ALL INDIA'
        GROUP BY brit_top_50_companies;
        
        Question: Calculate Num of Dealers for Britannia and PARLE PRODS for Cookies and ALL India in Dec-23?
        SQL Query: 
        SELECT brit_top_50_companies AS Company, SUM(Numerator) AS `Num of Dealers`
        FROM company_region
        WHERE Market = 'U+R'
            AND SL = 'SLY'
            AND KPI = 'Num of Dealers'
            AND brit_top_50_companies IN ('BRITANNIA INDS', 'PARLE PRODS')
            AND period = 'Dec-23'
            AND brit_seg3_670 = 'COOKIES'
            AND region = 'ALL INDIA'
        GROUP BY brit_top_50_companies;
        
        Question: Calculate MS for Britannia for SALT and ALL India in Dec-23?
        SQL Query: 
        SELECT brit_top_50_companies AS Company, SUM(Numerator/Denominator) AS MS
        FROM format_company
        WHERE Market = 'U+R'
            AND SL = 'SLY'
            AND KPI = 'MS'
            AND brit_top_50_companies LIKE 'BRITANNIA INDS'
            AND period = 'Dec-23'
            AND brit_seg3_670 = 'CRACKERS'
            AND brit_subseg = 'SALT'
            AND region = 'ALL INDIA'
        GROUP BY brit_top_50_companies;
        

        
        If you have additional questions or need to calculate metrics for multiple companies or segments, provide the necessary details in your question, and ensure your SQL query follows a similar format.

            
        ```<SCHEMA> {schema} </SCHEMA>```

        Conversation History: {conversation_history}

        Write only the SQL query without any additional text.

        Response Format:
            Question: {question}
            SQL Query:
        Additional Details or Follow-up Questions:
        "Could you specify the region for which you want to calculate the Market Share?"
        "Do you have a specific product category in mind for the calculation?"
        "Are there any particular columns in the dataset that you think might affect the Market Share calculation?"

    """

    prompt = ChatPromptTemplate.from_template(template=prompt_template)

    def get_schema(_):
        return db.get_table_info()

    return (
            RunnablePassthrough.assign(schema=get_schema)
            | prompt
            | llm
            | StrOutputParser()
    )

# Get response for user query
def get_response(user_query: str, db_name: str, conversation_history: list, llm):
    db = connect_database(db_name)
    sql_chain = get_sql_chain(db, llm)

    # Template for SQL response prompt
    prompt_template = """
        You are a senior data analyst. 
        Given the database schema details, question, SQL query, and SQL response, 
        write a natural language response for the SQL query.

        <SCHEMA> {schema} </SCHEMA>
        
        Conversation History: {conversation_history}
        SQL Query: <SQL> {sql_query} </SQL>
        Question: {question}
        SQL Response: {response}
        
        Response Format:
            SQL Query:
            Response:
    """

    prompt = ChatPromptTemplate.from_template(template=prompt_template)

    chain = (
            RunnablePassthrough.assign(sql_query=sql_chain).assign(
                schema=lambda _: db.get_table_info(),
                response=lambda vars: db.run(vars["sql_query"])
            )
            | prompt
            | llm
            | StrOutputParser()
    )

    return chain.invoke({
        "question": user_query,
        "conversation_history": conversation_history
    })

# Streamlit app
def main():
    st.set_page_config(page_title="SQL Chat", page_icon=":speech_balloon:")
    st.title("SQL Chat")

    # Select database
    db_name = st.selectbox("Select Database", list(db_uris))

    # Load LLM model
    llm = load_llm()

    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = [
            AIMessage(content="Hello! I am a SQL assistant. Ask me questions about your MYSQL database.")
        ]

    # Display conversation history
    for message in st.session_state.conversation_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)

    # User input for SQL query
    user_query = st.text_area("Enter your SQL query:", "", key="user_query")
    
    # Button to submit query
    if st.button("Ask"):
        response = get_response(user_query, db_name, st.session_state.conversation_history, llm)
        st.text("Response:")
        st.write(response)

if __name__ == "__main__":
    main()

