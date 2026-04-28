import sys
import gradio as gr
from privacy_filter_redactor import DecodingMode, PIIRedactor

# Initialize model immediately at module level
print(">>> Loading Privacy Filter Engine...", file=sys.stderr)
# We stay on CPU for the Hugging Face Free Tier
redactor = PIIRedactor(device="cpu")
print(f">>> Engine Ready on {redactor.device}", file=sys.stderr)

# Define Example Data
EXAMPLES = {
    "System Log (Short)": (
        "SYSTEM LOG: Error in worker node-7. Failed to authenticate using API key 'sk-live-51MzW2EJfGq8'. Action initiated by admin@tech-startup.io.",  # noqa: E501
        DecodingMode.BALANCED.value,
    ),
    "User Prompt (Short)": (
        "USER PROMPT: Hey, I'm Alice Smith from 123 Maple Ave. My social is 555-01-9999. Can you help me find my tax records from 2025?",  # noqa: E501
        DecodingMode.HIGH_RECALL.value,
    ),
    "Agent Output (Short)": (
        "AGENT OUTPUT: I've successfully processed the invoice for John Doe. Please call him at (555) 012-3456 to confirm receipt of the $1,200 payment.",  # noqa: E501
        DecodingMode.HIGH_PRECISION.value,
    ),
    "Customer Support Email (Long)": (
        "Subject: Urgent: Change of address and billing inquiry\n\n"
        "Dear Support Team,\n\n"
        "I am writing to update my contact information. My name is Michael O'Connor, "
        "and I recently moved from 789 Broadway, Vancouver, BC to 456 Oak Street, Seattle, WA 98101. "  # noqa: E501
        "Please update my records accordingly.\n\n"
        "Also, I noticed an unfamiliar charge on my credit card (ending in 4422). "
        "Could you please verify if this was for my premium subscription? "
        "You can reach me at m.oconnor@provider.net or call my mobile at +1 (206) 555-0188. "
        "My account number is ACCT-99210-XP.\n\n"
        "Thank you,\nMichael",
        DecodingMode.BALANCED.value,
    ),
    "Meeting Transcript (Long)": (
        "MEETING TRANSCRIPT - PROJECT X-RAY\n"
        "DATE: 2026-04-20\n"
        "PARTICIPANTS: Sarah Jenkins, David Chen\n\n"
        "Sarah: Hi David, thanks for joining. We need to finalize the vendor access for the new API.\n"  # noqa: E501
        "David: Sure Sarah. I've already shared the temporary credentials with the team. "
        "The master token is 'prod-key-8892-vbf-2231' and the backup is 'sec_9910_kmn_004'.\n"
        "Sarah: Got it. I'll make sure they are stored in the vault. By the way, the client "
        "representative, Robert Miller (robert.m@client-corp.com), asked if we can speed up the "
        "onboarding process. He lives in a different timezone, so his direct line is +44 20 7946 0958.\n"  # noqa: E501
        "David: I'll try to prioritize it. I'm also waiting for the final contract from their "
        "legal team at 10 Downing Street, London.",
        DecodingMode.HIGH_RECALL.value,
    ),
    "Internal Server Report (Long)": (
        "--- SECURITY AUDIT REPORT ---\n"
        "GENERATED: April 28, 2026\n"
        "ADMIN: Administrator (admin-ops@internal.cloud)\n\n"
        "The following anomalies were detected in the production environment:\n"
        "1. Multiple failed login attempts on account 'user_9921' from IP 192.168.1.1.\n"
        "2. Potential credential leak in the staging.log file. Found string 'password=admin12345!' "
        "associated with employee Jane Williams (jane.w@company.com).\n"
        "3. Database connection string exposed in cleartext: "
        "mongodb://dbuser:P@ssword99!@db.internal.cloud:27017/prod_db.\n"
        "4. Unauthorized access request for file 'C:\\Users\\BobSmith\\Documents\\TaxReturns.pdf'.\n"  # noqa: E501
        "Please investigate these issues immediately and notify the security officer.",
        DecodingMode.HIGH_PRECISION.value,
    ),
}

SELECT_PROMPT = "-- Select an Example --"


def load_example(title):
    """Returns the text and mode for a selected example title."""
    if not title or title == SELECT_PROMPT:
        return gr.update(), gr.update()
    return EXAMPLES[title]


def process_text(text, mode):
    """Main inference function."""
    if not text.strip():
        return "Please enter some text.", []

    redacted_text, entities = redactor.redact_with_details(text, mode=DecodingMode(mode.lower()))

    entity_data = [
        [ent.label, ent.text, f"{ent.start}:{ent.end}", f"{ent.score:.4f}"] for ent in entities
    ]

    return redacted_text, entity_data


# 2. Build the UI
with gr.Blocks(title="Privacy Filter Redactor") as demo:
    gr.Markdown("# Privacy Filter Redactor")
    gr.Markdown("Local PII detection and redaction using OpenAI's privacy-filter model.")

    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Input Text", placeholder="Paste text with PII here...", lines=12
            )

            example_dropdown = gr.Dropdown(
                choices=[SELECT_PROMPT] + list(EXAMPLES.keys()),
                value=SELECT_PROMPT,
                label="Try an Example",
            )

            mode_dropdown = gr.Dropdown(
                choices=[m.value for m in DecodingMode],
                value=DecodingMode.BALANCED.value,
                label="Decoding Mode",
            )

            submit_btn = gr.Button("Redact", variant="primary")

        with gr.Column():
            output_text = gr.Textbox(
                label="Redacted Result", lines=12, interactive=False, show_label=True
            )

    entity_table = gr.Dataframe(
        headers=["Category", "Text", "Offsets", "Confidence"],
        label="Detected Entities",
        interactive=False,
    )

    # Event Triggers
    example_dropdown.change(
        fn=load_example, inputs=[example_dropdown], outputs=[input_text, mode_dropdown]
    )

    submit_btn.click(
        fn=process_text, inputs=[input_text, mode_dropdown], outputs=[output_text, entity_table]
    )

if __name__ == "__main__":
    demo.launch()
