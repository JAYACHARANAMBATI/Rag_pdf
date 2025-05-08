
# 📚 Enhanced PDF RAG Assistant

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-00A67E?style=for-the-badge)
![Google-Gemini](https://img.shields.io/badge/Google%20Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-00A67E?style=for-the-badge)

An intelligent PDF analysis tool powered by Google Gemini, LangChain, and Qdrant that provides structured, context-aware responses to your document queries.

## 🎥 Demo Video

[![PDF RAG Assistant Demo](

https://github.com/user-attachments/assets/54aade83-f6af-4eb2-a511-be0744dd1858

)

*(Click the image above to watch the demo video)*

## ✨ Features

- **Smart Document Analysis**: Processes PDFs to understand content structure and relationships
- **Context-Aware Responses**: Adapts answer format based on question type (summaries, lists, comparisons, etc.)
- **Advanced Retrieval**: Uses Qdrant vector store with MMR search for diverse, relevant results
- **Conversational Interface**: Maintains chat history for natural interactions
- **Multi-Page Support**: Handles documents of varying lengths efficiently

## 🛠️ Technologies Used

- **Backend**:
  - [LangChain](https://python.langchain.com/) - Framework for building LLM applications
  - [Google Gemini](https://ai.google.dev/) - LLM for text generation and embeddings
  - [Qdrant](https://qdrant.tech/) - Vector similarity search engine

- **Frontend**:
  - [Streamlit](https://streamlit.io/) - For building interactive web apps

- **Other Tools**:
  - [PyPDF](https://pypi.org/project/pypdf/) - PDF text extraction
  - [RecursiveCharacterTextSplitter](https://python.langchain.com/docs/modules/data_connection/document_transformers/) - Document chunking

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pdf-rag-assistant.git
   cd pdf-rag-assistant
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file and add your Google API key:
   ```env
   GOOGLE_API_KEY=your_api_key_here
   ```

## 🏃‍♂️ Running the Application

1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your browser to `http://localhost:8501`

3. Upload a PDF document and start asking questions!

## � Requirements

- Python 3.8+
- Google API key (for Gemini access)
- Required Python packages (see `requirements.txt`)

## 📂 Project Structure

```
pdf-rag-assistant/
├── app.py                # Main application file
├── requirements.txt      # Python dependencies
├── README.md             # This documentation file
├── .env.example          # Environment variables template
└── assets/               # For demo screenshots/videos
    └── demo.mp4
```

## 🌟 Usage Examples

1. **Summarization**:
   - "Summarize the key points of this document"
   - "Give me a brief overview of chapter 3"

2. **Step-by-Step Instructions**:
   - "List the steps in the process"
   - "How do I perform X according to this document?"

3. **Comparisons**:
   - "Compare the advantages and disadvantages of X and Y"
   - "What's the difference between method A and method B?"

4. **Explanations**:
   - "Why is this concept important?"
   - "Explain how this system works"

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📧 Contact

Your Name - Ambati Jaya Charan

Project Link: https://www.linkedin.com/in/ambati-jaya-charan-901052254/
```
