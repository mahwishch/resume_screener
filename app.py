import gradio as gr
import os
import llama_index


from resume_screener_pack.llama_index.packs.resume_screener.base import ResumeScreenerPack as llamapack



# Job description and criteria
meta_jd = """\
Meta is embarking on the most transformative change to its business and technology in company history, and our Machine Learning Engineers are at the forefront of this evolution. By leading crucial projects and initiatives that have never been done before, you have an opportunity to help us advance the way people connect around the world.
The ideal candidate will have industry experience working on a range of recommendation, classification, and optimization problems. You will bring the ability to own the whole ML life cycle, define projects and drive excellence across teams. You will work alongside the world’s leading engineers and researchers to solve some of the most exciting and massive social data and prediction problems that exist on the web.\
"""

criteria = [
    "2+ years of experience in one or more of the following areas: machine learning, recommendation systems, pattern recognition, data mining, artificial intelligence, or related technical field",
    "Experience demonstrating technical leadership working with teams, owning projects, defining and setting technical direction for projects",
    "Bachelor's degree in Computer Science, Computer Engineering, relevant technical field, or equivalent practical experience.",
]

# Set OpenAI API key
os.environ['OPENAI_API_KEY'] = os.getenv("openai_key")    #userdata.get('openAI2024')

# Define the resume screening function
def screen_resume(resume):
    resume_screener = llamapack(
        job_description=meta_jd,
        criteria=criteria,
    )
    response = resume_screener.run(resume_path=resume)

    decisions = []
    for cd in response.criteria_decisions:
        decisions.append(f"### CRITERIA DECISION\n{cd.reasoning}\nDecision: {cd.decision}\n")

    overall_reasoning = f"#### OVERALL REASONING #####\n{str(response.overall_reasoning)}\nOverall Decision: {str(response.overall_decision)}"

    return "\n".join(decisions) + "\n" + overall_reasoning

# Create the Gradio interface
iface = gr.Interface(
    fn=screen_resume,
    inputs=gr.File(file_types=[".pdf"]),
    outputs="text",
    title="Resume Screener",
    description="Upload a PDF resume to screen it according to the provided job description and criteria."
)

# Launch the interface
iface.launch(auth=("atomcamp", "12345678"))
