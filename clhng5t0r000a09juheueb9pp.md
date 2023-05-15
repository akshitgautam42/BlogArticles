---
title: "How I Built a PDF Chat Application using  Langchain 🦜️🔗and OpenAI"
datePublished: Sun May 14 2023 13:23:32 GMT+0000 (Coordinated Universal Time)
cuid: clhng5t0r000a09juheueb9pp
slug: how-i-built-a-pdf-chat-application-using-langchain-and-openai
tags: machine-learning, beginners, streamlit, beginners-learningtocode-100daysofcode, langchain

---

I am excited to share my journey of building a PDF Chat application using Langchain and Python.In this blog post, I'll take you through the process and share the insights I gained along the way.Let's get started:

## **Setting up the Environment: A Journey of Dependencies**

Using pip command install the following dependencies:

```python
pip install langchain==0.0.154
pip install PyPDF2==3.0.1
pip install python-dotenv==1.0.0
pip install streamlit==1.18.1
pip install faiss-cpu==1.7.4
```

Once everything is installed insert your OpenAI API Key in the .env file as below:

```python
OPENAI_API_KEY=""
```

## **Creating the GUI: Unleashing My Creativity**

With my environment set up, I delved into creating the graphical user interface (GUI) for the PDF Chat application. I decided to use Streamlit, a powerful Python library, for its simplicity and ease of use. I wanted to create an interface that was visually appealing and intuitive for users.

Import the streamlit library using

```python
import streamlit as st
```

Then in the main function use streamlit to build the GUI:

```python
st.set_page_config(page_title="Ask your PDF")
st.header("Ask your PDF 💬")

pdf=st.file_uploader("Upload your PDF",type="pdf")
```

Once the GUI is done it will look like this:

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1684059047584/16382a4d-6ffa-4d83-a5e1-5855b28839d2.png align="center")

## **Parsing the PDF: Unveiling the Secrets**

Next comes up the task of extracting text from the uploaded PDF. I turned to the PyPDF2 library for help.

```python
from PyPDF2 import PdfReader


pdf_reader=PdfReader(pdf)
text =""
for page in pdf_reader.pages:
    text+=page.extract_text()
```

## **Chunking the Text: Breaking It Down**

Handling the entire text as a single entity would have been impractical. So, I explored the concept of text chunking. I leveraged Langchain's text splitter functionality to divide the extracted text into smaller, manageable chunks. This approach enabled me to perform efficient semantic search and retrieve relevant information based on user queries.

```python
from langchain.text_splitter import CharacterTextSplitter

text_splitter=CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function = len
        )
chunks = text_splitter.split_text(text)
```

## **Power of Embeddings: Capturing Semantic Meaning**

Using the OpenAI embeddings and FAISS (Facebook AI Similarity Search) I created the knowledgebase.

```python
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

 embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks,embeddings)
```

## **Answering Questions: Unleashing the Power of Langchain**

The real magic happened when I integrated Langchain into the application.

To show user input I used streamlit below:

```python
user_question = st.text_input("Ask a question:")
```

And used LangChain to use Question Answer based chaining also used the get\_callback functionality to determine how much OpenAI API is costing me for the answers:

```python
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
docs=knowledge_base.similarity_search(user_question)
llm=OpenAI()
chain=load_qa_chain(llm,chain_type="stuff")
with get_openai_callback() as cb:
    response =chain.run(input_documents=docs,question=user_question)
    print(cb)
```

Congrats you made it!

You have now built a Full Fledged application using the power of LangChain.Now go ahead and try combining the pieces of the code together yourself and build the application.

Preview of the application built:

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1684070548226/6dde54ba-500e-430c-9b69-1daf1f75d8b2.png align="center")

I am attaching the full code but try once yourself before using it.

[Ask Your PDF](https://github.com/akshitgautam42/AskYourPDF)