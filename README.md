# LLM-powered Chatbot with RAG

This project builds an intelligent chatbot using Large Language Models (LLM) combined with Retrieval Augmented Generation (RAG) technology.

## Overview

This chatbot leverages the power of LLMs and RAG techniques to generate accurate and contextually relevant responses. It utilizes MongoDB as a database, Langchain for natural language processing, Google's Gemini API for multimodal capabilities, and the Transformers library for implementing language models.

## Key Features

- LLM-based response generation with high accuracy
- Relevant information retrieval using RAG
- Efficient data storage and retrieval with MongoDB
- Powerful natural language processing with Langchain
- Multimodal capabilities (text and image processing) with Gemini
- Language model deployment and fine-tuning using Transformers

## Installation

1. Clone the repository:

```bash
git clone git@github.com:dungca1512/llm_retrieval_rag.git
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up MongoDB and update connection information in the configuration file

4. Configure API keys for Gemini and other services (if required)

## Usage

To run the chatbot:

1. Navigate to the project directory:

```bash
cd llm_retrieval_rag
```

2. Start the chatbot:

```bash
python query.py
```

3. Follow the on-screen prompts to interact with the chatbot

## Configuration

Modify the `config.yaml` file to customize:

- MongoDB's connection settings
- LLM model selection
- RAG parameters
- Gemini API settings

## Project Structure

```
llm-retrieval-rag/
│
├── manage.py            # Entry point of the application
├── config.yaml          # Configuration file
├── requirements.txt     # Project dependencies
│
├── src/
│   ├── llm/             # LLM integration
│   ├── rag/             # RAG implementation
│   ├── database/        # MongoDB interactions
│   ├── langchain/       # Langchain utilities
│   └── gemini/          # Gemini API integration
│
├── models/              # Pretrained and fine-tuned models
│
└── tests/               # Unit and integration tests
```

## Contributing

We welcome contributions to this project. Please feel free to submit issues or pull requests for any improvements or bug fixes.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Langchain](https://github.com/hwchase17/langchain)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [MongoDB](https://www.mongodb.com/)
- [Google Gemini API](https://cloud.google.com/vertex-ai/docs/generative-ai/multimodal/overview)

## Contact

Email: dungca1512@gmail.com

Project Link: [https://github.com/dungca1512/llm_retrieval_rag](https://github.com/dungca1512/llm_retrieval_rag)