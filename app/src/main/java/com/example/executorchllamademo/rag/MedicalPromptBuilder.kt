package com.example.executorchllamademo.rag

/** Builds the ChatML-formatted conversational RAG prompt (system prompt, background knowledge, recent history, user message) fed to the model. */
object MedicalPromptBuilder {

    const val SYSTEM_PROMPT = """You are a calm, warm, and supportive conversational assistant for people experiencing emotional or psychological distress.

You are not a doctor, psychologist, psychotherapist, or emergency service. You do not diagnose, prescribe medication, or replace professional care.
Your goal is to have a natural, ongoing conversation, not to give the same support message every time.

CRITICAL INSTRUCTION FOR CLINICAL KNOWLEDGE:
You have access to background medical notes about the user.
Use this to understand what they are going through and tailor your empathy.

Answer style:
- Be warm, human, simple, and empathetic.
- Respond directly to what the user just said.
- Continue the conversation from the previous turn
- Keep the answer short (2 to 5 sentences).
- Avoid repeating same phrases
- Avoid generic reassurance.
- Suggest at most ONE small practical step if useful.
- Ask at most ONE gentle follow-up question. Do not end every message with a question.
- Only mention emergency services if the user explicitly talks about self-harm, suicide.

Do not repeat the same opening sentence in consecutive replies.

/no_think"""

    fun buildConversationalRagPrompt(
        userMessage: String,
        retrievedContexts: List<String>,
        patientContext: String? = null,
        recentConversation: String? = null
    ): String {
        
        val backgroundBlock = buildString {
            if (retrievedContexts.isNotEmpty() || !patientContext.isNullOrBlank()) {
                append("BACKGROUND KNOWLEDGE\n")
                if (!patientContext.isNullOrBlank()) {
                    append("Patient Summary: $patientContext\n")
                }
                if (retrievedContexts.isNotEmpty()) {
                    append("Relevant Clinical Notes:\n")
                    retrievedContexts.forEach { append("- $it\n") }
                }
                append("\n")
            }
        }

        val historyBlock = if (!recentConversation.isNullOrBlank()) {
            "RECENT DIALOGUE\n$recentConversation\n\n"
        } else ""

        val userPromptContent = """
$backgroundBlock$historyBlock CURRENT USER MESSAGE 
$userMessage

Instructions:
- Reflect on the user's message using your background knowledge to understand their mental/physical state.
- Answer naturally without sounding like a medical document.
""".trimIndent()

        return """
<|im_start|>system
$SYSTEM_PROMPT<|im_end|>
<|im_start|>user
$userPromptContent<|im_end|>
<|im_start|>assistant
""".trimStart()
    }
}