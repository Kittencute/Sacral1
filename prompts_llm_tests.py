intent_llm_prompt = (
    "You are an information extractor for queries.\n"
    "Extract and label only the following fields from the user's question.\n"
    "Use this exact format, and leave any field blank if not found:\n\n"
    "course_name: \n"
    "program_name: \n"
    "keywords: \n\n"
    "Rules:\n"
    "- A course code has exactly 3 letters followed by 3 digits (e.g., DVA222, cdt406).\n"
    "- A program code has exactly 3 letters followed by 2 digits (e.g., CCV20, dat21).\n"
    "- 'course_name' is any phrase that sounds like a course (e.g., 'Lärande system').\n"
    "- 'program_name' contains the word 'program' or 'programmet' (e.g., 'Sjuksköterskeprogrammet').\n"
    "- 'keywords' reflect what the user is asking about (e.g., overview, prerequisites, examination).\n"
    "- Do **not** include course or program codes — that is handled separately.\n"
    "- Do **not** include codes in the course or program names.\n"
    "- Do **not explain** your output. Just return the field values."
)

    courses = []
    seen = set()
    current_year = None

    # Split by "Årskurs X"
    blocks = re.split(r"(Årskurs\s+\d+)", text)
    i = 0

    while i < len(blocks):
        block = blocks[i].strip()

        # Detect and update current year
        if block.startswith("Årskurs"):
            year_match = re.match(r"Årskurs\s+(\d+)", block)
            if year_match:
                current_year = int(year_match.group(1))
            i += 1
            continue

        # Skip empty text chunks
        if not block:
            i += 1
            continue

        # Extract categories and course blobs
        category_block_re = re.finditer(
            r"([\w/åäöÅÄÖ\s\-\–]+?):\s*((?:[^:]+?\d+(?:[.,]\d+)?\s*hp\s*)+)",
            block,
            re.UNICODE,
        )
        for match in category_block_re:
            category = match.group(1).strip()
            course_blob = match.group(2)

            # Extract individual courses
            course_re = re.finditer(r"([^,]+?),\s*(\d+(?:[.,]\d+)?)\s*hp", course_blob)
            for c in course_re:
                course_name = c.group(1).strip()
                hp = float(c.group(2).replace(",", "."))

                # Use lowercase tuple to avoid dupes
                key = (course_name.lower(), hp, current_year, category.lower())
                if key in seen:
                    continue
                seen.add(key)

                courses.append(Course(
                    name=course_name,
                    hp=hp,
                    year=current_year,
                    category=category
                ))

        i += 1

    return courses