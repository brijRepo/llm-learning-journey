import ollama
from json_utils import extract_json
import json

RESUME_SCHEMA = {
    "name": "string",
    "email": "string",
    "phone": "string or null",
    "skills": ["skill1", "skill2"],
    "experience": [
        {
            "company": "string",
            "role": "string",
            "experience": "string",
            "responsibilities": ["resp1", "resp2"]
        }
    ],
    "education": [
        {
            "degree": "string",
            "institution": "string",
            "year": "string or null"
        }
    ]
}

def parsed_resume(resume_text):

    prompt = f"""Parse this resume and extract information into JSON.
    
    Resume Text:
    {resume_text}

    Extract into this JSON structure:
    {json.dumps(RESUME_SCHEMA, indent=2)}

    Rules:
    1. Return ONLY valid JSON (no markdown, no extra text)
    2. Extract ALL skills mentioned
    3. List jobs in order (most recent first)
    4. Extract 2-4 key responsibilities per job
    5. Use null for missing fields (not "unknown" or "N/A")

    JSON:"""

    response = ollama.chat(
        model='llama3.2:3b',
        messages=[{'role': 'user', 'content': prompt}]
    )
    
    # Extract JSON
    return extract_json(response['message']['content'])

SAMPLE_RESUME = """
JOHN DOE
Email: john.doe@email.com | Phone: (555) 123-4567

EXPERIENCE

Senior Software Engineer | Tech Corp | 2020-Present
- Led development of microservices architecture serving 1M users
- Reduced API latency by 40% through caching optimization
- Mentored team of 5 junior engineers
- Implemented CI/CD pipeline with GitHub Actions

Software Engineer | StartupXYZ | 2018-2020
- Built RESTful APIs using Python and Flask
- Developed frontend dashboards with React
- Managed PostgreSQL databases

EDUCATION

BS Computer Science | State University | 2018

SKILLS
Python, JavaScript, Docker, AWS, PostgreSQL, React, Git, CI/CD
"""

def main():
    print("Resume Parser")
    print("=" * 70)
    print("\nParsing sample resume...\n")

    parsed = parsed_resume(SAMPLE_RESUME)

    if parsed:
        print("Parsed Resume:")
        print(json.dumps(parsed, indent=2))

        with open("parsed_resume.json", "w") as f:
            json.dump(parsed, f, indent=2)

        print("\nSaved to parsed_resume.json")
        
        # Show summary
        print("\n" + "=" * 70)
        print("Summary:")
        print("=" * 70)
        print(f"Name: {parsed.get('name', 'N/A')}")
        print(f"Email: {parsed.get('email', 'N/A')}")
        print(f"Skills: {len(parsed.get('skills', []))} skills found")
        print(f"Jobs: {len(parsed.get('experience', []))} positions")
        print(f"Education: {len(parsed.get('education', []))} degrees")
    else:
        print("Failed to parse resume")


if __name__ == "__main__":
    main()