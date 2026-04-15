"""LLM prompt templates for the churn retention agent."""

from langchain_core.prompts import ChatPromptTemplate

EXPLANATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a senior customer success analyst at a major telecom company. "
            "Explain in 3–4 plain-English sentences why this customer is at risk of churning, "
            "referencing their specific feature values (e.g. short tenure, high charges, "
            "month-to-month contract). Never use machine learning jargon. "
            "Tone: professional, empathetic, data-grounded. "
            "Output: one paragraph only.",
        ),
        (
            "human",
            "Customer profile:\n{customer_features}\n\n"
            "Churn probability: {churn_probability}\n"
            "Risk level: {risk_level}\n"
            "Top risk factors: {risk_factors}\n\n"
            "Retrieved retention context:\n{retrieved_context}\n\n"
            "Explain why this customer is at risk.",
        ),
    ]
)

RECOMMENDATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert retention specialist at a telecom company. "
            "Given the customer profile, churn probability, explanation, and retrieved "
            "retention strategies, write exactly 3 numbered, personalized, actionable "
            "recommendations that a customer service agent could execute on a real call. "
            "Each recommendation should be 1–2 sentences. "
            "You must reference the customer's specific situation. "
            "Output: numbered list only, no preamble.",
        ),
        (
            "human",
            "Customer profile:\n{customer_features}\n\n"
            "Churn probability: {churn_probability}\n"
            "Risk level: {risk_level}\n"
            "Explanation: {explanation}\n\n"
            "Retrieved strategies:\n{retrieved_context}\n\n"
            "Provide exactly 3 retention recommendations.",
        ),
    ]
)

SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Generate a single executive-summary sentence (under 25 words) "
            "capturing the retention action priority for this customer. "
            "Output: exactly one sentence.",
        ),
        (
            "human",
            "Customer profile:\n{customer_features}\n\n"
            "Churn probability: {churn_probability}\n"
            "Risk level: {risk_level}\n"
            "Recommendations: {recommendations}\n\n"
            "Write the executive summary.",
        ),
    ]
)
