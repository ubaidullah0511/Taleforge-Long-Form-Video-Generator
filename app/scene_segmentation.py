from app.config import settings
from app.llm_client import generate_json
from app.models import Scene

PROMPT = """Split the following video script into logical scenes, where each \
scene is one distinct visual idea/beat. Return a JSON object with a single \
key "scenes" whose value is an array of objects with keys "scene" (1-based \
int) and "text" (the scene's script text, verbatim substring of the input).

Script:
\"\"\"{script}\"\"\"
"""


def split(script: str) -> list[Scene]:
    data = generate_json(PROMPT.format(script=script), settings.llm_text_model)
    return [Scene(**item) for item in data["scenes"]]
