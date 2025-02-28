import os
import tempfile
import streamlit as st
from pypdf import PdfReader
from langchain_community.graphs import Neo4jGraph
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MISTRAL_MODEL = "mixtral-8x7b-32768"  

def get_neo4j_connection():
    """Create and verify Neo4j connection using LangChain's Neo4jGraph"""
    try:
        graph = Neo4jGraph(
            url=st.session_state.neo4j_uri,
            username=st.session_state.neo4j_user,
            password=st.session_state.neo4j_pass
        )
        graph.query("RETURN 1 AS test") 
        return graph
    except Exception as e:
        st.error(f"Connection failed: {str(e)}")
        return None

def process_pdf(file_path):
    """Process PDF and return non-empty chunks"""
    try:
        reader = PdfReader(file_path)
        text = " ".join([page.extract_text() or "" for page in reader.pages])
        chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
        return [chunk for chunk in chunks if chunk.strip()]
    except Exception as e:
        st.error(f"PDF processing error: {str(e)}")
        return []

def create_text_index(graph):
    """Create full-text search index using LangChain connection"""
    try:
        graph.query("""
            CREATE FULLTEXT INDEX documentChunks IF NOT EXISTS
            FOR (n:DocumentChunk) ON EACH [n.content]
        """)
    except Exception as e:
        st.error(f"Index creation failed: {str(e)}")

def get_llm_response(query, context,api):
    """Generate response using Chat Groq and Mistral model"""
    try:
       
        llm = ChatGroq(model_name=MISTRAL_MODEL, temperature=0.7,api_key=api)
        
        
        prompt = ChatPromptTemplate.from_template("""
            You are a helpful assistant. Use the following context to answer the question:
            
            Context:
            {context}
            
            Question:
            {query}
            
            Answer:
        """)
        
        # Create chain
        chain = prompt | llm | StrOutputParser()
        
        # Generate response
        return chain.invoke({"query": query, "context": context})
    except Exception as e:
        st.error(f"LLM error: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Graph RAG with Mistral", layout="wide")
    
    
    if 'processed' not in st.session_state:
        st.session_state.update({
            'processed': False,
            'neo4j_uri': "bolt://localhost:7687",
            'neo4j_user': "neo4j",
            'neo4j_pass': "password",
            'groq_key': ""  
        })

    
    with st.sidebar:
        st.header("Configuration")
        st.session_state.neo4j_uri = st.text_input("Neo4j URI", st.session_state.neo4j_uri)
        st.session_state.neo4j_user = st.text_input("Neo4j User", st.session_state.neo4j_user)
        st.session_state.neo4j_pass = st.text_input("Neo4j Password", type="password", 
                                                  value=st.session_state.neo4j_pass)
        st.session_state.groq_key = st.text_input("Groq API Key", type="password")
        
        if st.button("Test Connection"):
            graph = get_neo4j_connection()
            if graph:
                st.success("✅ Neo4j connection successful!")
                create_text_index(graph)

   
    st.title("📚 Graph RAG with Mistral 🤖")
    
    uploaded_files = st.file_uploader(
        "Upload PDFs", 
        type="pdf", 
        accept_multiple_files=True
    )

    if uploaded_files and st.button("Process Documents"):
        graph = get_neo4j_connection()
        if not graph:
            return

        create_text_index(graph)
        
        with st.status("Processing...", expanded=True) as status:
            for file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp.write(file.getvalue())
                    tmp_path = tmp.name
                
                try:
                    chunks = process_pdf(tmp_path)
                    if not chunks:
                        st.warning(f"No valid text found in {file.name}")
                        continue
                    
                    
                    graph.query("""
                        MERGE (d:Document {name: $doc_name})
                        WITH d
                        UNWIND $chunks AS chunk
                        CREATE (d)-[:HAS_CHUNK]->(:DocumentChunk {
                            content: chunk,
                            timestamp: datetime()
                        })
                        """,
                        {"doc_name": file.name, "chunks": chunks}
                    )
                    
                    st.write(f"✅ Processed {file.name} ({len(chunks)} chunks)")
                finally:
                    os.unlink(tmp_path)
            
            status.update(label="Complete!", state="complete")
            st.session_state.processed = True

    if st.session_state.get('processed'):
        query = st.text_input("Ask about your documents:")
        if query and st.session_state.groq_key:
            try:
                graph = get_neo4j_connection()
                if not graph:
                    return
                
                
                results = graph.query("""
                    CALL db.index.fulltext.queryNodes("documentChunks", $query)
                    YIELD node, score
                    RETURN node.content AS text, score
                    LIMIT 5
                    """,
                    {"query": query}
                )
                
                
                context = "\n\n".join([r["text"] for r in results])
                
                
                response = get_llm_response(query, context,st.session_state.groq_key)
                
                if response:
                    st.subheader("Answer")
                    st.markdown(response)
                    
                    with st.expander("See retrieved context"):
                        for i, record in enumerate(results, 1):
                            st.markdown(f"**Context {i}** (Score: {record['score']:.2f})")
                            st.write(record["text"])
                            st.divider()
                        
            except Exception as e:
                st.error(f"Search error: {str(e)}")

if __name__ == "__main__":
    main()