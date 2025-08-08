### Project: Textual Summarization Of Videos in Indic Languages

---

#### Description:
This project aims to provide textual summarization of videos in Indic languages using various technologies and APIs.

---

#### Backend Application:
The backend application (`app.py`) is written in Flask, a lightweight WSGI(Web Server Gateway Interface) web application framework in Python. All the required functions for summarization are present in the `main.py` file.

---

#### Technologies and APIs Used:
- **Summarization**: We are using ChatGPT 3.5 Turbo, a state-of-the-art natural language processing model developed by OpenAI, for the summarization portion.
- **Translation**: For translation, we are utilizing Bhashini APIs to facilitate translation of text into Indic languages.
- **Text-to-Speech Translation**: For text-to-speech translation, we are employing gTTS (Google Text-to-Speech), a Python library and CLI tool to interface with Google Translate's text-to-speech API.

---

#### How to Run the Backend:
Follow these steps to run the backend:

1. **Clone the Backend Branch**: Clone the backend branch of the repository.

2. **Install Dependencies**: Navigate to the cloned repository directory and run the following command to install the required dependencies using pip: `pip install -r requirements.txt`


3. **Set Up API Keys and Authorization Tokens**: Open the `main.py` file and input your OpenAI API key and Bhashini's authorization token in the appropriate placeholders.

4. **Run the Application**: Once the dependencies are installed and API keys are set up, run the Flask application by executing the following command: `python app.py`


---
#### Exploring Various Approaches:
We have explored various different approaches in this project. For detailed code implementations and experimentation, refer to the following Colab notebooks:
1. [Approach 1: Generating Transcript of videos present in different Indic languages and converting transcript back to english](https://colab.research.google.com/drive/1-l_oEfw5XK648of40NxBeNxV3TfPhPT6?usp=sharing)
2. [Approach 2: NLP Training](https://colab.research.google.com/drive/1F7JO2SLnNVLxPey65Qi3xmOLnlK_LhOQ?usp=sharing)
3. [Approach 3: Using different models for summarization](https://colab.research.google.com/drive/1Bb0B_cBjW6QwBMgv4KDXKNUbB89Sgp8f?usp=sharing)
4. [Approach 4: Using different models for Translation and generating audio of translated Summary](https://colab.research.google.com/drive/1bhnv7gfPqnu6ylz-8dh5WtFZtykq-479?usp=sharing)

---
#### Note:
Ensure that you have appropriate permissions and access to the required APIs before running the application.

---

### Future Enhancements
- Add option to download summaries as PDF or text.
- Integrate more Indic languages.
- Include sentiment analysis of video transcripts.
- Improve summarization for videos longer than 10 minutes.

---

### Contributing
We welcome contributions! Please:
1. Fork this repository.
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m "feat: describe feature"`
4. Push to branch and open a Pull Request.