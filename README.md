AI Story Generator with Image


Project Objective


The objective of this project is to create an AI-powered application that takes a story idea (short prompt or concept) as input and produces a cohesive narrative along with supporting images for each part of the story. This tool blends natural language generation (NLG) and text-to-image synthesis to deliver an engaging, illustrated storytelling experience. It’s designed for creatives, writers, educators, or anyone who wants to visualize a story from just an idea.

Functional Requirements & Core Features

User Input: Story Idea

A single-sentence or short-paragraph input from the user, e.g., "A young girl finds a secret door in her grandmother's attic."
Optionally, allow selecting:
Genre (e.g., fantasy, sci-fi, mystery, comedy)
Tone (e.g., dark, lighthearted, epic)
Target audience (kids, teens, adults)
Narrative Generation

The system should expand the idea into a multi-part narrative, typically:
Introduction / Setting
Conflict / Rising Action
Climax
Resolution
Can be structured into 3–5 story segments or "scenes."
Text should be coherent, stylistically consistent, and tailored to the genre/tone.
Image Generation

For each scene or paragraph of the story, generate an illustrative image using a text-to-image model.
The image prompt can be auto-extracted or generated from the scene’s text.
Output Format

Display the story scene by scene or page by page, each with:
A block of narrative text
A corresponding generated image
Setup and Running
To set up and run this project, follow these steps:

Clone the repository:
cd AI-Story-Generator-Hackathon

 pip install -r requirements.txt

   streamlit run guvi.py
