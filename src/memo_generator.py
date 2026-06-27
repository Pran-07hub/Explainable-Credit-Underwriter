from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# pyrefly: ignore [missing-import]
from langchain_google_genai import ChatGoogleGenerativeAI
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="meta-llama/Meta-Llama-3-8B-Instruct", task="text-generation")
model = ChatHuggingFace(llm=llm)
# model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

def prompt_form(risk_score, risk_factor_text, protective_factor_text):
    prompt = f"""
    You are a Senior Credit Risk Auditor at a commercial bank.

    Your task is to write a concise underwriting memo for a loan officer.

    Rules:

    * Maximum 120 words.
    * Use a professional banking tone.
    * Do not mention SHAP, machine learning, AI, or model explanations.
    * Refer to factors in business language rather than column names whenever possible.
    * Explain both risk factors and mitigating factors.
    * End with a clear recommendation: Approve, Manual Review, or Reject.
    * Do not use bullet points.

    Applicant Default Risk Score: {risk_score:.2%}

    Primary Risk Drivers:
    {risk_factor_text}

    Protective Factors:
    {protective_factor_text}

    Write the memo in exactly this structure:

    CREDIT REVIEW MEMO

    Risk Assessment:
    <3-4 sentence assessment>

    Recommendation:
    <Approve(if risk < 18%)/ Manual Review (if 18% < risk < 45%) / Reject(if risk > 45%) with one sentence justification>
    """
    return prompt

def generate_memo(risk_score, risk_factors, protective_factors):
    risk_factor_strings = [
        f"- {item['feature']} ({item['description'] or 'No description available'}): {item['value']}"
        for item in risk_factors
    ]
    protective_factor_strings = [
        f"- {item['feature']} ({item['description'] or 'No description available'}): {item['value']}"
        for item in protective_factors
    ]
    prompt = prompt_form(risk_score, risk_factor_strings, protective_factor_strings)
    response = model.invoke(prompt)
    return response.content