import streamlit as st                            # For Web Interface (Front-End)
from pdfminer.high_level import extract_text      # To Extract Text from Resume PDF
from sentence_transformers import SentenceTransformer      # To generate Embeddings of text
from sklearn.metrics.pairwise import cosine_similarity     # To get Similarity Score of Resume and Job Description
from groq import Groq                             # API to use LLM's
import re                                         # To perform Regular Expression Functions
from dotenv import load_dotenv                    # Loading API Key from .env file
import os


# Load environment variables from .env
load_dotenv()

# Fetch the key from the environment
api_key = os.getenv("GROQ_API_KEY")


#  Session States to store values 
if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False

if "resume" not in st.session_state:
    st.session_state.resume=""

if "job_desc" not in st.session_state:
    st.session_state.job_desc=""



# Title of the Project, change according to your style
st.title("SmartCV 📝")



# <------- Defining Functions ------->

# Function to extract text from PDF
def extract_pdf_text(uploaded_file):
    try:
        extracted_text = extract_text(uploaded_file)
        return extracted_text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return "Could not extract text from the PDF file."


# Function to calculate similarity 
def calculate_similarity_bert(text1, text2):
    ats_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')      # Use BERT or SBERT or any model you want
    # Encode the texts directly to embeddings
    embeddings1 = ats_model.encode([text1])
    embeddings2 = ats_model.encode([text2])
    
    # Calculate cosine similarity without adding an extra list layer
    similarity = cosine_similarity(embeddings1, embeddings2)[0][0]
    return similarity


def get_report(resume,job_desc):
    client = Groq(api_key=api_key)

    # Change the prompt to get the results in your style
    prompt = f"""
# Context:
You are an AI Resume Analyzer. You will be given a candidate's resume and the job description for a specific role.

# Instructions:
- Carefully analyze the candidate’s resume based on the job description.
- Evaluate the resume for each of the following 6 categories:
    1. 💼 Skills Match
    2. 📚 Experience Relevance
    3. 🎓 Educational Background
    4. 📜 Certifications or Courses
    5. 🧪 Projects or Contributions
    6. 🖋 Overall Presentation & Formatting

- For *each category*:
    - Begin with: <b>[Emoji] Category: [Emoji Mark] Score/5</b>
    - Add a detailed *explanation* for the evaluation.
    - If there's anything missing or that could be improved, include a line like:
      👉 <i>Improvement:</i> [Your Suggestion]

- Use these marks:
    - ✅ = Aligned
    - ❌ = Not aligned
    - ⚠ = Unclear / Partial match

- Finish with:
    <b>🔢 Overall AI Score: X/5</b>  
    <b>💡 Suggestions to improve your resume:</b>
    Mention any additional overall suggestions that could improve the candidate’s chances.

# Inputs:
Candidate Resume: {resume}
---
Job Description: {job_desc}

# Output Format:
<b>📄 Resume Analysis for the role of [Job Role]</b><br><br>

<b>1. 💼 Skills Match: ✅ 4/5</b><br>
✅ Strong alignment with Python, SQL, and ML tools required.<br>
👉 <i>Improvement:</i> Could add experience in cloud platforms like AWS if required in JD.<br><br>

<b>2. 📚 Experience Relevance: ❌ 2/5</b><br>
❌ Projects exist but lack full-time professional experience in this domain.<br>
👉 <i>Improvement:</i> Add more job-relevant roles or internships if possible.<br><br>

<b>3. 🎓 Educational Background: ✅ 5/5</b><br>
✅ Degree matches the required qualifications.<br><br>

<b>4. 📜 Certifications or Courses: ⚠ 3/5</b><br>
⚠ Certifications exist but are only partially relevant.<br>
👉 <i>Improvement:</i> Include certifications in required tools/skills mentioned in JD.<br><br>

<b>5. 🧪 Projects or Contributions: ✅ 4.5/5</b><br>
✅ Projects are strong but lack deployment or collaboration details.<br>
👉 <i>Improvement:</i> Add outcome metrics or user impact of the project.<br><br>

<b>6. 🖋 Overall Presentation & Formatting: ⚠ 3/5</b><br>
⚠ Well structured but inconsistent font sizes and spacing.<br>
👉 <i>Improvement:</i> Use clear headings, consistent formatting, and bullet alignment.<br><br>

<b>🔢 Overall AI Score: 3.75/5</b><br><br>

<b>💡 Suggestions to improve your resume:</b>
<ul>
  <li>✅ Highlight key accomplishments with metrics (e.g. "improved accuracy by 20%").</li>
  <li>✅ Tailor your summary to match the job's key keywords.</li>
  <li>✅ Add more experience or responsibilities aligned with the role.</li>
  <li>✅ Use a cleaner format with consistent spacing and fonts.</li>
  <li>✅ Mention any tools/frameworks listed in the job description, even if self-learned.</li>
</ul>
"""

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content

def extract_scores(text):
    # Regular expression pattern to find scores in the format x/5, where x can be an integer or a float
    pattern = r'(\d+(?:\.\d+)?)/5'
    # Find all matches in the text
    matches = re.findall(pattern, text)
    # Convert matches to floats
    scores = [float(match) for match in matches]
    return scores




# <--------- Starting the Work Flow ---------> 

# Displays Form only if the form is not submitted
if not st.session_state.form_submitted:
    with st.form("my_form"):

        # Taking input a Resume (PDF) file 
        resume_file = st.file_uploader(label="Upload your Resume in PDF format", type="pdf")

        # Taking input Job Description
        st.session_state.job_desc = st.text_area("Please provide the job description for the position you're applying to.:",placeholder="Job Description...")

        # Form Submission Button
        submitted = st.form_submit_button("Analyze")
        if submitted:

            #  Allow only if Both Resume and Job Description are Submitted
            if st.session_state.job_desc and resume_file:
                st.info("Extracting Information")

                st.session_state.resume = extract_pdf_text(resume_file)      # Calling the function to extract text from Resume

                st.session_state.form_submitted = True
                st.rerun()                 # Refresh the page to close the form and give results

            # Donot allow if not uploaded
            else:
                st.warning("Please Upload both Resume and Job Description to analyze")


if st.session_state.form_submitted:
    score_place = st.info("Generating Scores...")

    # Call the function to get ATS Score
    ats_score = calculate_similarity_bert(st.session_state.resume,st.session_state.job_desc)

    col1,col2 = st.columns(2,border=True)
    with col1:
        st.write("Few ATS uses this score to shortlist candidates, Similarity Score:")
        st.subheader(str(ats_score))

    # Call the function to get the Analysis Report from LLM (Groq)
    report = get_report(st.session_state.resume,st.session_state.job_desc)
    

    # Calculate the Average Score from the LLM Report
    report_scores = extract_scores(report)                 # Example : [3/5, 4/5, 5/5,...]
    avg_score = sum(report_scores) / (5*len(report_scores))  # Example: 2.4


    with col2:
        st.write("Total Average score according to our AI report:")
        st.subheader(str(avg_score))
    score_place.success("Scores generated successfully!")


    st.subheader("AI Generated Analysis Report:")

    # Displaying Report 
    st.markdown(f"""
            <div style='text-align: left; background-color: #D1C4E9; padding: 20px; border-radius: 12px; margin: 10px 0; font-size: 16px; line-height:1.6;'>
                {report}
            </div>
            """, unsafe_allow_html=True)
    
    # Download Button
    st.download_button(
        label="Download Report",
        data=report,
        file_name="report.txt",
        icon=":material/download:",
        )
    

# <-------------- End of the Work Flow --------------->