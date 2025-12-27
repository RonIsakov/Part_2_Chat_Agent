
You are given 4 days to complete this assessment. For this assignment, you have access to the following Azure OpenAI resources:

- Document Intelligence for Optical Character Recognition (OCR)
- GPT-4o and GPT-4o Mini as Large Language Models (LLMs)
- ADA 002 for text embeddings

All required resources have already been deployed in Azure. There is no need to create additional resources for this assignment.

The necessary Azure credentials have been included in the email containing this assignment. Please refer to these credentials for accessing the pre-deployed resources.

## **IMPORTANT NOTE:** Use only the native Azure OpenAI SDK library, not LangChain or other frameworks.

## Repository Contents

- **phase2_data**: This folder contains:
  - HTML files that serve as the knowledge base for Part 2 of the home assignment

## Part 2: Microservice-based ChatBot Q&A on Medical Services

### Task
Develop a microservice-based chatbot system that answers questions about medical services for Israeli health funds (Maccabi, Meuhedet, and Clalit) based on user-specific information. The system should be capable of handling multiple users simultaneously without maintaining server-side user memory.

### Core Requirements

1. **Microservice Architecture**
   - Implement the chatbot as a stateless microservice using FastAPI or Flask.
   - Handle multiple concurrent users efficiently.
   - Manage all user session data and conversation history client-side (frontend).

2. **User Interface**
   - Develop a frontend using **Gradio** or **Streamlit**.
   - Implement two main phases: User Information Collection and Q&A.

3. **Azure OpenAI Integration**
   - Utilize the Azure OpenAI client library for Python.
   - Implement separate prompts for the information collection and Q&A phases.

4. **Data Handling**
   - Use provided HTML files provided in the 'phase2_data' folder as the knowledge base for answering questions.

5. **Multi-language Support**
   - Implement support for Hebrew and English. 

6. **Error Handling and Logging**
   - Implement comprehensive error handling and validation.
   - Create a logging system to track chatbot activities, errors, and interactions.

### Detailed Specifications

#### User Information Collection Phase
Collect the following user information:
- First and last name
- ID number (valid 9-digit number)
- Gender
- Age (between 0 and 120)
- HMO name (מכבי | מאוחדת | כללית)
- HMO card number (9-digit)
- Insurance membership tier (זהב | כסף | ארד)
- Provide a confirmation step for users to review and correct their information.

**Note:** This process should be managed exclusively through the LLM, avoiding any hardcoded question-answer logic or form-based filling in the UI


#### Q&A Phase
- Transition to answering questions based on the user's HMO and membership tier.
- Utilize the knowledge base from provided HTML files.

#### State Management
- Pass all necessary user information and conversation history with each request to maintain statelessness.

### Evaluation Criteria

1. Microservice Architecture Implementation
2. Technical Proficiency (Azure OpenAI usage, data processing)
3. Prompt Engineering and LLM Utilization
4. Code Quality and Organization
5. User Experience
6. Performance and Scalability
7. Documentation
8. Innovation
9. Logging and Monitoring Implementation

### Submission Guidelines
1. Provide source code via GitHub.
2. Include setup and run instructions.

**Good luck! For any questions, feel free to contact me.**

Dor Getter.
