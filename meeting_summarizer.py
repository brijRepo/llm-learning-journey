import ollama
from json_utils import extract_json
import json

MEETING_SCHEMA = {
    "meeting_title": "string",
    "date": "YYYY-MM-DD",
    "attendees": ["person1", "person2"],
    "key_points": ["point1", "point2"],
    "action_items": [
        {
            "task": "string",
            "owner": "string",
            "deadline": "string or null"
        }
    ],
    "decisions_made": ["decision1", "decision2"],
    "next_meeting": "string or null"
}

def summarize_meeting(transcript):

    prompt = f"""Summarize this meeting transcript into structured notes.

    Transcript:
    {transcript}

    Extract into this JSON format:
    {json.dumps(MEETING_SCHEMA, indent=2)}

    Instructions:
    1. Extract meeting title from context
    2. Identify all attendees by name
    3. List 3-5 key discussion points
    4. Extract action items with owners and deadlines
    5. Note any decisions made
    6. Capture next meeting if mentioned

    Use null or [] for missing information.
    Return ONLY valid JSON.

    JSON:"""

    response = ollama.chat(
        model='llama3.2:3b',
        messages=[{'role': 'user', 'content': prompt}]
    )
    
    return extract_json(response['message']['content'])

SAMPLE_TRANSCRIPT = """
Meeting: Q1 Planning - Product Team
January 15, 2025

Attendees: Sarah (PM), Mike (Eng Lead), Jenny (Designer)

Sarah: Let's discuss Q1 priorities. Focus on new dashboard feature.

Mike: Agreed. Estimate 3 weeks with 2 engineers. Start by Feb 1st.

Jenny: I'll have designs ready by Friday, January 22nd. Will share in Slack.

Sarah: Mike, can you write the technical spec this week?

Mike: Yes, spec done by Thursday, January 18th.

Sarah: Target launch is February 25th. Mike will assign John and Lisa.

Jenny: Users reported mobile app is slow. Should we prioritize?

Sarah: Keep that for Q2. Focus on dashboard now.

Mike: Next meeting same time next Tuesday, January 23rd at 2pm?

Sarah: Perfect. I'll send recap email today.
"""

def main():

    print("Meeting Notes Summarizer")
    print("=" * 70)
    
    # Summarize
    notes = summarize_meeting(SAMPLE_TRANSCRIPT)

    if notes:
        print("\nMEETING NOTES")
        print("=" * 70)

        if notes.get('meeting_title'):
            print(f"\nTitle: {notes['meeting_title']}")
        
        if notes.get('date'):
            print(f"Date: {notes['date']}")

        if notes.get('attendees'):
            print(f"\nAttendees ({len(notes['attendees'])}):")
            for person in notes['attendees']:
                print(f"   • {person}")

        if notes.get('key_points'):
            print(f"\nKey Points:")
            for i, point in enumerate(notes['key_points'], 1):
                print(f"   {i}. {point}")

        if notes.get('action_items'):
            print(f"\nAction Items:")
            for item in notes['action_items']:
                task = item.get('task', 'N/A')
                owner = item.get('owner', 'Unassigned')
                deadline = item.get('deadline', 'No deadline')
                print(f"   • {task}")
                print(f"    Owner: {owner} | Due: {deadline}")
        
        if notes.get('decisions_made'):
            print(f"\nDecisions:")
            for i, decision in enumerate(notes['decisions_made'], 1):
                print(f"    {i}.{decision}")
        
        if notes.get('next_meeting'):
            print(f"\nNext Meeting: {notes['next_meeting']}")

        with open("meeting_notes.json", "w") as f:
            json.dump(notes, f, indent=2)
        
        print(f"\nSaved to meeting_notes.json")

        print("\n" + "=" * 70)
        print("ACTION ITEMS CHECKLIST")
        print("=" * 70)
        if notes.get('action_items'):
            for item in notes['action_items']:
                task = item.get('task', '')
                owner = item.get('owner', '')
                deadline = item.get('deadline', '')
                print(f"[ ] {task} (@{owner}) - {deadline}")
    
    else:
        print("Failed to parse meeting")


if __name__ == "__main__":
    main()
