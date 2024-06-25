from typing import List
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_community.document_loaders import (
    PyMuPDFLoader,
)
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.indexes import SQLRecordManager, index
from langchain.schema import Document
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableConfig
from langchain.callbacks.base import BaseCallbackHandler
import pymupdf4llm
from langchain.text_splitter import MarkdownTextSplitter
from llms import get_multimodal_llm, get_graq_model
import chainlit as cl
import tempfile
import os

embeddings_model = OpenAIEmbeddings()

# Use relative path to the Data directory
CODE_STORAGE_PATH = os.path.join(os.path.dirname(__file__), '../Data')
# define a persistent directory for ChromaDB:
PERSIST_DIRECTORY = os.path.join(os.path.dirname(__file__), '../src/chroma_db')

def process_python_files(pdf_storage_path: str):
    pdf_directory = Path(pdf_storage_path)
    docs = []
    splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=0)

    for pdf_path in pdf_directory.glob("*.pdf"):
        md_text = pymupdf4llm.to_markdown(str(pdf_path))  # get markdown for all pages
        chunks = splitter.create_documents([md_text])
        
        # Add source metadata to each chunk
        for chunk in chunks:
            chunk.metadata['source'] = str(pdf_path)
        
        docs += chunks

    # Check if the Chroma database already exists
    if os.path.exists(PERSIST_DIRECTORY):
        # Load the existing database
        doc_search = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings_model)
    else:
        # Create a new database and persist it
        doc_search = Chroma.from_documents(docs, embeddings_model, persist_directory=PERSIST_DIRECTORY)
        doc_search.persist()

    namespace = "chromadb/my_documents"
    record_manager = SQLRecordManager(
        namespace, db_url="sqlite:///record_manager_cache.sql"
    )
    record_manager.create_schema()

    index_result = index(
        docs,
        record_manager,
        doc_search,
        cleanup="incremental",
        source_id_key="source",
    )

    print(f"Indexing stats: {index_result}")

    return doc_search

doc_search = process_python_files(CODE_STORAGE_PATH)
model = get_graq_model()  # ChatOpenAI(model_name="gpt-4", streaming=True)


@cl.on_chat_start
async def on_chat_start():
    template = """Use the following piece of source information based on the experiment name  generate a detailed configuration file for the experiment. 
    The configuration should include all relevant parameters, their enable/disable status, and their values. 
    based on user requirment add and reamove relevant parameters.
    Use the following format stucture out Don not write any thing else:
    
    # Name of the experiment = experiment_name

    # Example parameters (Add or remove as needed)
    
    # Alpha  (Internal 'alphash')
    ena_Alpha=[true/false]
    val_Alpha=[value]

    # Enable/Disable Ay1
    ena_Ay1=[true/false]
    val_Ay1=[value]

    # Enable/Disable Ay2
    ena_Ay2=[true/false]
    val_Ay2=[value]

    # Enable/Disable Ay3
    ena_Ay3=[true/false]
    val_Ay3=[value]
    
    # Enable/Disable Az1
    ena_Az1=[true/false]
    val_Az1=[value]

    # Enable/Disable Az2
    ena_Az2=[true/false]
    val_Az2=[value]

    # Enable/Disable Az3
    ena_Az3=[true/false]
    val_Az3=[value]

    # Enable/Disable Base
    ena_Base=[true/false]
    val_Base=[value]

    # Enable/Disable Twinned
    ena_Twinned=[true/false]
    val_Twinned=[true/false]

    # Enable/Disable WAXS
    ena_WAXS=[true/false]
    val_WAXS=[true/false]

    # Enable/Disable CBInterior
    ena_CBInterior=[true/false]
    val_CBInterior=[type value]

    # CBParticle  (Particle type selection)
    ena_CBParticle=[true/false]
    val_CBParticle=[type  value]

    # Enable/Disable CBPeak
    ena_CBPeak=[true/false]
    val_CBPeak=[type value]

    # Enable/Disable Azi
    ena_Azi=[true/false]
    val_Azi=[value]

    # Enable/Disable BFactor
    ena_BFactor=[true/false]
    val_BFactor=[value]

    # Enable/Disable Ceff
    ena_Ceff=[true/false]
    val_Ceff=[value]

    # Enable/Disable Ceffcyl
    ena_Ceffcyl=[true/false]
    val_Ceffcyl=[value]

    # Enable/Disable Dbeta
    ena_Dbeta=[true/false]
    val_Dbeta=[value]

    # DebyeWaller  (Also called Displacement [nm])
    ena_DebyeWaller=[true/false]
    val_DebyeWaller=[value]

    # Det  (Distance Sample - Detector [m])
    ena_Det=[true/false]
    val_Det=[value]

    # Enable/Disable Dist
    ena_Dist=[true/false]
    val_Dist=[value]

    # DomainSize  (Radial domain size [nm])
    ena_DomainSize=[true/false]
    val_DomainSize=[value]

    # Enable/Disable PeakPar
    ena_PeakPar=[true/false]
    val_PeakPar=[value]

    # PixelNoX  (Number of horizontal detector pixel)
    ena_PixelNoX=[true/false]
    val_PixelNoX=[value]

    # PixelNoY  (Number of vertical detector pixel)
    ena_PixelNoY=[true/false]
    val_PixelNoY=[value]

    # PixelX  (Width of one detector pixel [mm])
    ena_PixelX=[true/false]
    val_PixelX=[value]

    # PixelY  (Height of one detector pixel [mm])
    ena_PixelY=[true/false]
    val_PixelY=[value]

    # Qmax  (Qmax preset from user [nm-1])
    ena_Qmax=[true/false]
    val_Qmax=[value]

    # QmaxData  (Use the Qmax from the data)
    ena_QmaxData=[true/false]
     val_QmaxData=[true/false]

    # QmaxPreset  (Use the Qmax provided here)
    ena_QmaxPreset=[true/false]
    val_QmaxPreset=[true/false]

    # Radius  (Inner radius)
    ena_Radius=[true/false]
    val_Radius=[value]

    # Radiusi  (Outer radius)
    ena_Radiusi=[true/false]
    val_Radiusi=[value]

    # Enable/Disable Rho
    ena_Rho=[true/false]
    val_Rho=[value]

    # Enable/Disable Sigma
    ena_Sigma=[true/false]
    val_Sigma=[value]

    # Wavelength  (Wavelength [nm])
    ena_Wavelength=[true/false]
    val_Wavelength=[value]

    # HKLmax  (Number of iterations in the h,k,l-loops)
    ena_HKLmax=[true/false]
    val_HKLmax=[value]

    # Enable/Disable I0
    ena_I0=[true/false]
    val_I0=[value]

    # LType  (Lattice type selection)
    ena_LType=[true/false]
    val_LType=[type value]

    # Enable/Disable Length
    ena_Length=[true/false]
    val_Length=[value]

    # Enable/Disable Ordis
    ena_Ordis=[true/false]
    val_Ordis=[type value]

    # Enable/Disable P1
    ena_P1=[true/false]
    val_P1=[value]

    # Enable/Disable DebyeScherrer
    ena_DebyeScherrer=[true/false]
    val_DebyeScherrer=[true/false]

    # Enable/Disable RBPara
    ena_RBPara=[true/false]
    val_RBPara=[true/false]

    # RotAlpha  (Internal 'alpha')
    ena_RotAlpha=[true/false]
    val_RotAlpha=[value]

    # SigX  (editdom1)
    ena_SigX=[true/false]
    val_SigX=[value]

    # SigY  (editdom2)
    ena_SigY=[true/false]
    val_SigY=[value]

    # SigZ  (editdom3)
    ena_SigZ=[true/false]
    val_SigZ=[value]

    # Enable/Disable SigmaL
    ena_SigmaL=[true/false]
    val_SigmaL=[value]

    # Enable/Disable Ax1
    ena_Ax1=[true/false]
    val_Ax1=[value]

    # Enable/Disable Ax2
    ena_Ax2=[true/false]
    val_Ax2=[value]

    # Enable/Disable Ax3
    ena_Ax3=[true/false]
    val_Ax3=[value]

    # Enable/Disable acpl
    ena_acpl=[true/false]
    val_acpl=[value]

    # Enable/Disable bcpl
    ena_bcpl=[true/false]
    val_bcpl=[value]

    # Enable/Disable ifluc
    ena_ifluc=[true/false]
    val_ifluc=[value]

    # Enable/Disable iso
    ena_iso=[true/false]
    val_iso=[value]

    # Enable/Disable phi
    ena_phi=[true/false]
    val_phi=[value]

    # Enable/Disable reff
    ena_reff=[true/false]
    val_reff=[value]

    # Enable/Disable rfluc
    ena_rfluc=[true/false]
    val_rfluc=[value]

    # Enable/Disable rotPhi
    ena_rotPhi=[true/false]
    val_rotPhi=[value]

    # Enable/Disable rotTheta
    ena_rotTheta=[true/false]
    val_rotTheta=[value]

    # Enable/Disable theta
    ena_theta=[true/false]
    val_theta=[value]

    # uca  (Unit cell dimension a [nm])
    ena_uca=[true/false]
    val_uca=[value]

    # ucalpha  (Unit cell rotation alpha [°])
    ena_ucalpha=[true/false]
    val_ucalpha=[value]

    # ucb  (Unit cell dimension b [nm])
    ena_ucb=[true/false]
    val_ucb=[value]

    # ucbeta  (Unit cell rotation beta [°])
    ena_ucbeta=[true/false]
    val_ucbeta=[value]

    # ucc  (Unit cell dimension c [nm])
    ena_ucc=[true/false]
    val_ucc=[value]

    # ucgamma  (Unit cell rotation gamma [°])
    ena_ucgamma=[true/false]
    val_ucgamma=[value]

    # Enable/Disable ucn1
    ena_ucn1=[true/false]
    val_ucn1=[value]

    # Enable/Disable ucn2
    ena_ucn2=[true/false]
    val_ucn2=[value]

    # Enable/Disable ucn3
    ena_ucn3=[true/false]
    val_ucn3=[value]

    # Enable/Disable ucpsi
    ena_ucpsi=[true/false]
    val_ucpsi=[value]

    # GridPoints  (Half of the size of each image dimension)
    ena_GridPoints=[true/false]
    val_GridPoints=[value]

    # Enable/Disable BeamPos
    ena_BeamPos=[true/false]
    val_BeamPosX=[value]  # -GridPoints .. +GridPoints
    val_BeamPosY=[value]  # -GridPoints .. +GridPoints

    # Enable/Disable Generate PNG
    ena_generatePNG=[true/false]

    # Number of Images
    val_numimg=[value]

    # Output Path
    val_outPath=[path]
    
    
    
    
    If you don't know the answer, just say that you don't know, don't try to generate any random answer from your own

    Context:{context}
    Question:{question}

    Only return the configuration file in the specified format and nothing else
    helpful answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    retriever = doc_search.as_retriever()

    runnable = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable
    msg = cl.Message(content="")

    class PostMessageHandler(BaseCallbackHandler):
        """
        Callback handler for handling the retriever and LLM processes.
        Used to post the sources of the retrieved documents as a Chainlit element.
        """

        def __init__(self, msg: cl.Message):
            super().__init__()
            self.msg = msg
            self.sources = set()
            self.response_text = ""

        def on_retriever_end(self, documents, *, run_id, parent_run_id, **kwargs):
            for d in documents:
                source_page_pair = (d.metadata['source'], d.metadata['page'])
                self.sources.add(source_page_pair)

        def on_llm_end(self, response, *, run_id, parent_run_id, **kwargs):
            try:
                print(f"LLM response generations: {response.generations}")  # Debugging statement
                if isinstance(response.generations[0], list):
                    self.response_text = response.generations[0][0].text
                else:
                    self.response_text = response.generations[0].text
                print(f"Retrieved response text: {self.response_text}")  # Debugging statement
            except Exception as e:
                print(f"Error retrieving response text: {e}")
            if len(self.sources):
                sources_text = "\n".join([f"{source}#page={page}" for source, page in self.sources])
                self.msg.elements.append(
                    cl.Text(name="Sources", content=sources_text, display="inline")
                )

    post_message_handler = PostMessageHandler(msg)

    async with cl.Step(type="run", name="QA Assistant"):
        async for chunk in runnable.astream(
            message.content,
            config=RunnableConfig(callbacks=[
                cl.LangchainCallbackHandler(),
                post_message_handler
            ]),
        ):
            await msg.stream_token(chunk)

    await msg.send()

    # Generate .txt file with the response text and provide download link
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmpfile:
        tmpfile.write(post_message_handler.response_text.encode('utf-8'))
        tmpfile_path = tmpfile.name
        #print(f"Response text written to file: {tmpfile_path}")  # Debugging statement

    elements = [
        cl.File(
            name="experiment.txt",
            path=tmpfile_path,
            display="inline",
        ),
    ]

    await cl.Message(
        content="download file from here👇 ", elements=elements
    ).send()