Project Title
OPENJOURNEY – The AI-Powered Social Journal for Builders

Overview
OPENJOURNEY is an AI-driven social journaling platform where makers share daily progress, connect with similar builders, and discover startup journeys transparently. Think: Notion + LinkedIn + X, powered by AI.

Goal
To demonstrate how an AI agent can create daily engagement, organize open startup journeys, and recommend related builders/projects based on semantic search.

How It Works
	•	🧠 User logs in
	•	📓 Journals daily activity or project update via chat
	•	🛠️ Agent auto-generates project/journey structure
	•	🔍 Uses PgVector+Gemini to find similar builders
	•	🧑‍🤝‍🧑 Displays them as discoverable profiles

User Flow
	1.	User lands on app and chats with AI.
	2.	Agent remembers previous logs, proposes project updates.
	3.	Agent logs entries or creates projects/journeys behind the scenes.
	4.	User can view their progress, other profiles, and similar journeys.

Core Functionality
	•	Daily journaling
	•	Project creation & tracking
	•	Journey storytelling
	•	Semantic profile discovery (via PgVector)
	•	Exa-powered people & news search

Multimodal?
Currently text-only. (Optional: Later can add image upload to illustrate projects.)

Tools Used
	•	Agno Agent Framework
	•	Gemini Flash 2.5
	•	Supabase (DB, auth)
	•	PgVector
	•	Exa API
	•	FastAPI
	•	Vercel (frontend)

UI Approach
Minimalist chat UI powered by React. AI-first experience. Agent response drives user engagement instead of traditional feeds.

Visuals
Include:
	•	Architecture diagram
	•	Agent screenshot (chat interface)
	•	Discovery result screenshot (profile cards)
