# How to Interact with the Experiential Network GPT

Welcome! This guide helps you speak naturally and lets the GPT map your words into stress and content values (1–5) based on the context of your network.

## 1. Start Your Network

• Simply introduce yourself:

* “Hi, I’m ready to begin my experiential network.”
* GPT will call `create_user_sheet` and confirm your unique sheet ID.

## 2. Add Experiences (Nodes) Naturally

• Describe how you feel, and GPT will infer:

* **Stress vs. Content**: Words like “anxious,” “overwhelmed,” or “worried” become stress nodes; words like “calm,” “relaxed,” or “joyful” become content nodes.
* **Relative Intensity (1–5)**: GPT gauges intensity by comparing to your previous entries. For example:

  * “I’m a bit nervous” → stress = 2
  * “I’m extremely anxious” → stress = 5
  * “I feel somewhat at ease” → content = 3

• **Optional Explicit Intensity**:

* If you want to override, simply say: “Set that to a 4 out of 5.”

GPT will then call `upsert_node` with the determined **Energy** value.

## 3. Link and Contextualize (Connections)

• Show relationships naturally:

* “My anxiety comes from upcoming exams.”
* “My relief follows deep breathing.”

GPT maps these to `upsert_connection`, assigning positive or negative actions and default weights.

## 4. Check Your Network Snapshot

• Ask in plain language:

* “What does my network look like?”
* “Show me where my stress is highest.”

GPT calls `fetch_experiential_network` and summarizes:

* Lists top stress and content nodes
* Displays your current stress vs. content ratio

## 5. Refine and Reflect

• Modify intensity by speaking:

* “Reduce my stress about exams to a 2.”
* “Increase my relief after walking to a 5.”

• Add new nodes on the fly:

* “I felt a little proud after finishing the task.”

GPT updates via `upsert_node` and fetches again for feedback.

## 6. Celebrate Balance

• When your content ratio exceeds 70%, GPT will call `check_and_join` and congratulate you:

* “🎉 You’ve reached a healthy balance and joined the collective!”

## 7. Speak Without Code

• No JSON or commands needed—just talk:

* ✔️ “I’m overwhelmed by work; please help calm me.”
* ❌ Don’t write `{"name":"upsert_node",…}`

## 8. Ask for Personalized Guidance

• Request targeted tips:

* “What can I do right now to lower my top stress node?”

GPT uses your real-time network to tailor suggestions.

---

Speak freely—GPT will handle the mapping, function calls, and updates, letting you focus on your emotional journey.
