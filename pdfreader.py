from typing import TypedDict, Union, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage,ToolMessage, AIMessage 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
load_dotenv()   
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7,
    max_output_tokens=1024,
    top_p=0.95,                    
    top_k=40,
    google_api_key=os.environ.get("GOOGLE_API_KEY")
)
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-1.5-flash")
pdf= "BAH Dantzig Transportation problem (1).pdf"  
if not os.path.exists(pdf):
    raise FileNotFoundError(f"File wasnt found")
 # Path to your PDF file 
loader= PyPDFLoader(pdf) 
docs = loader.load()  # Load the PDF documents
for i, doc in enumerate(docs):
    print(f".....{i+1} Result")
    print(f"....context: {doc.page_content}")

splitter= RecursiveCharacterTextSplitter(
    chunk_size=1000,    
    chunk_overlap=200
)
pages_split= splitter.split_documents(docs) 

vectorstore= Chroma.from_documents(
    documents=pages_split,  
    embedding=embeddings, 
    persist_directory="vectorstore"   
    # Specify the directory to persist the vector store
)
print(f"Vector store created with {len(vectorstore)} documents.")
retriever= vectorstore.as_retriever(
    search_type="similarity",    
    search_kwargs={"k": 3}  # Retrieve top 3 similar documents
)
@tool 
def retriever_tool(query: str) -> str:
    """This tool retrieves the information from the dantzigs transportation problem and returns the result """
    docs = retriever.invoke(query) 
    if not docs:
        return "No relevant documents found."    
    results=[] 
    for i, doc in enumerate(docs): 
        results.append(f"Document {i+1}:\n{doc.page_content}\n") 
    return "\n\n".join(results) 
tools = [retriever_tool]  # List of tools to be used by the model
llm_with_tools = model.bind_tools(tools)  # Bind the tools to the model
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]  # Define the state schema
def should_continue(state:AgentState): 
    """Check if the conversation should continue."""
    result= state["messages"][-1]  
    return "continue" if not result.tool_calls else "end"

def llm_call(state:AgentState)->AgentState:
    """calls the llm binded with the tools and returns the updated state.""" 
    system_prompt= SystemMessage(
        content="You are a intelligent ai assistant that answers question about the dantzigs transportation problem. Use the retriever tool to get relevant information from the documents.You are allowed to look up to sone information before answering the question. Please use the tools correctly and update the documents states after modification.Cite the sources of the information you provide." 

    ) 
    messages= list(state["messages"]) + [system_prompt]  # Add the system prompt to the messages
    response = llm_with_tools.invoke(messages)  
    return {'messages':response.content }

      # Convert to a list to avoid mutability issues
tools_dict={tool.name: tool for tool in tools}  # Create a dictionary of tools for easy access
def process_node(state: AgentState) -> AgentState:
    """execute the llm call and return the updated state."""
    tool_call= state["messages"][-1].tool_calls 
    results=[] 
    for tool_calls in tool_call:
        print (f"Tool call: {tool_calls.name} with args: {tool_call.args}")
        if  not tool_calls.name in tools_dict:
            print(f"Tool {tool_calls.name} not found.")
            result="Tool not found."
        else:
            result= tools_dict[tool_calls.name].invoke(tool_calls.args).get('query') # Call the tool with the provided arguments
        results.append(ToolMessage(tool_call_id=tool_calls.id, name= tool_calls.name , content=result))  # Append the tool result to the messages 
    print("tool execution eocmplete") 
    return {"messages":  results} 
 # Return the updated state with the tool results
graph = StateGraph(AgentState) 
graph.add_node("llm_call", llm_call)  # Add the LLM call node
graph.add_node("process_node", process_node) 
graph.add_edge(START, "llm_call")  # Add an edge from START to the LLM call node
graph.add_conditional_edges(
    "llm_call", 
    should_continue,
    { 
        True: "process_node",  # If the conversation should continue, go to the process node
        False: END  # If the conversation should end, go to END
    }
)
graph.add_edge("process_node", "llm_call") 
graph.set_entry_point("llm_call")  
 # Se t the entry point to the LLM call node 
agent = graph.compile() 
  # Compile the graph to create the agent
def print_stream(stream): 
    print("Stream output:") 
    while True:
     user_input = input("whats your question  ")  
     # Get user input 
     if user_input.lower() == "exit":
        break 
     messages= [HumanMessage(content=user_input)]  # Create a list of messages with the user input
     result= agent.invoke({"messages": messages} 
                         , stream=True)  
     # Invoke the agent with the messages
     print("AI:", result["messages"][-1].content) 
     # Print the AI response content 
