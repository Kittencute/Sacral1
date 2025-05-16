import json
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional
import re
import difflib
from typing import List, Dict, Tuple
from itertools import combinations

file_path = "mdu_data_url/program/programs.jsonl"
course_path = "mdu_data_url/course/courses.jsonl"

# === Data Structures ===

@dataclass
class Course:
    name: str
    hp: float
    year: int
    category: str = "Okänd"
    code: str = ""
    metadata: dict = None
    prerequisites: str = ""
    huvudomrade: str = "Okänt"
    
def split_giltig_fran(value: str):
    parts = re.split(r"\s+", value.strip())
    if len(parts) == 2:
        period = parts[0]
        try:
            year = int(parts[1])
            return period, year
        except ValueError:
            return None, None
    return None, None

def find_program_by_name(file_path: str, program_name: str) -> Optional[dict]:
    base_name_match = re.match(r"^(.*?)(\d{4})?$", program_name.strip())
    if base_name_match:
        name_base = base_name_match.group(1).strip()
        target_year = base_name_match.group(2)
    else:
        name_base = program_name.strip()
        target_year = None

    latest_program = None
    latest_score = -1

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            full_name = item.get("name", "").strip()
            if name_base.lower() not in full_name.lower():
                continue

            period, year = split_giltig_fran(item.get("giltig från", ""))
            if year is None:
                continue

            # If user specified a year in the name, filter to match it
            if target_year and str(year) != target_year:
                continue

            score = year + (0.5 if period.lower().startswith("höst") else 0.0)
            if score > latest_score:
                latest_program = item
                latest_score = score

    return latest_program
  
def extract_courses(text: str) -> List[Course]:
    courses = []
    seen = set()
    text += "\nÅrskurs 999"
    year_blocks = re.split(r"(Årskurs\s+\d+)", text)
    current_year = None

    i = 0
    while i < len(year_blocks):
        part = year_blocks[i].strip()
        year_match = re.match(r"Årskurs\s+(\d+)", part)
        if year_match:
            current_year = int(year_match.group(1))
            i += 1
            continue

        if current_year is None or not part:
            i += 1
            continue

        # Track last seen category
        last_category = "Okänt"

        # Match sequences of (optional category:) followed by course(s) with hp
        tokens = re.findall(r"((?:[\w/åäöÅÄÖ:\-\–\s]+?)?)\s*([^:,\n]+?),?\s*(\d+(?:[.,]\d+)?)\s*hp", part, re.UNICODE)
        for cat_or_blank, name, hp in tokens:
            hp_val = float(hp.replace(",", "."))
            cat = cat_or_blank.strip().rstrip(":")

            if ":" in cat_or_blank:
                last_category = cat
            elif cat:
                # if it's just more text but no colon, it's likely part of the name
                name = f"{cat} {name}"
                cat = last_category
            else:
                cat = last_category

            key = (name.lower(), current_year, cat.lower())
            if key in seen:
                continue
            seen.add(key)

            courses.append(Course(
                name=name.strip(),
                hp=hp_val,
                year=current_year,
                category=cat
            ))

        i += 1

    return courses

def parse_giltig_fran(term: str) -> float:
    match = re.match(r"(Hösttermin|Vårtermin)\s+(\d{4})", term)
    if match:
        return int(match.group(2)) + (0.5 if match.group(1).lower().startswith("höst") else 0.0)
    return 0.0  

def build_course_code_lookup(course_file: str) -> dict:
    lookup = {}
    with open(course_file, "r", encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            name = item.get("name", "").strip().lower()
            code = item.get("kurskod", "").strip().upper()
            if name and code:
                lookup[name] = item  # store full course dict
    return lookup
        
def format_courses_by_year(courses: List[Course]) -> List[str]:
    grouped = defaultdict(lambda: defaultdict(list))
    for course in courses:
        grouped[course.year][course.category].append(course)

    prereq_map = build_prerequisite_map(courses)

    course_blocks = []  # Each course entry as a block of lines (strings)

    for year in sorted(grouped):
        for category in grouped[year]:
            for course in grouped[year][category]:
                block = []
                block.append(f"<<< COURSE: {course.name} >>>")
                block.append(f"Code: {course.code or 'N/A'}")
                block.append(f"Name: {course.name}")
                block.append(f"Category: {category}")
                block.append(f"Year: {course.year}")
                block.append(f"Credits: {course.hp} hp")

                key = course.code or course.name
                dependents = prereq_map.get(key, [])
                if dependents:
                    block.append(f"{course.code} is needed for:")
                    for target in dependents:
                        block.append(f"* [{target.code}] {target.name} ({target.hp} hp)")

                block.append("<<< END COURSE >>>")
                course_blocks.append("\n".join(block))

    return course_blocks 
        
def build_prerequisite_map(courses: List[Course]) -> dict:
    reverse_map = defaultdict(list)

    for target in courses:
        prereq_text = target.prerequisites.lower()
        for candidate in courses:
            if candidate.name.lower() in prereq_text:
                key = candidate.code or candidate.name
                reverse_map[key].append(target)

    return reverse_map
        
def summarize_categories(courses: List[Course], target_total: float = 300.0):
    totals = defaultdict(float)
    computed_total = 0.0

    for course in courses:
        totals[course.category] += course.hp
        computed_total += course.hp

    print("\n--- Total hp per kategori ---\n")
    for category, hp in sorted(totals.items(), key=lambda x: -x[1]):
        percent = (hp / target_total) * 100 if target_total else 0
        percent_str = f"{percent:.1f}".replace(".", ",") 
        print(f"{category}\n{percent_str} % ({hp:.1f} hp)\n")

    print(f"Program total: {computed_total:.1f} hp (av {target_total:.1f} hp)\n")
    
def can_replace(target: Course, candidate_set: List[Course], dependents: List[Course]) -> bool:
    for dep in dependents:
        if target.name.lower() in dep.prerequisites.lower():
            for alt in candidate_set:
                if alt.name.lower() in dep.prerequisites.lower():
                    break
            else:
                return False
    return True

def find_replacement_combinations(
    target: Course,
    all_courses_data: dict,
    current_program_course_names: set,
    dependents: List[Course],
    max_extra_hp: float = 7.5
) -> List[Tuple[List[dict], float]]:
    target_hp = target.hp
    target_level = target.metadata.get("utbildningsnivå", "")
    target_domains = set()
    if target.huvudomrade and target.huvudomrade.strip().lower() != "okänt":
        target_domains = set(x.strip().lower() for x in target.huvudomrade.split(","))
    else:
        target_domains = {target.category.strip().lower()}

    # Filter viable candidates
    candidates = []
    for course_dict in all_courses_data.values():
        name = course_dict.get("name", "").strip()
        if name.lower() in current_program_course_names:
            continue
        if course_dict.get("utbildningsnivå", "") != target_level:
            continue
        domains = set(x.strip().lower() for x in course_dict.get("huvudområde(n)", "").split(","))
        if not domains & target_domains:
            continue
        try:
            hp = float(course_dict.get("omfattning", "0").replace(",", ".").replace(" hp", ""))
        except ValueError:
            continue
        course_dict["_parsed_hp"] = hp
        candidates.append(course_dict)

    viable = []
    for r in range(1, 4):  # Try 1-3 replacement combos
        for combo in combinations(candidates, r):
            total_hp = sum(c["_parsed_hp"] for c in combo)
            if total_hp < target_hp:
                continue
            if total_hp - target_hp > max_extra_hp:
                continue
            # Optionally, check for downstream compatibility here
            viable.append((list(combo), total_hp))
    return sorted(viable, key=lambda x: (abs(x[1] - target_hp), len(x[0])))
        
def load_all_courses(course_file: str) -> List[dict]:
    courses = []
    with open(course_file, "r", encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            courses.append(item)
    return courses

def collect_replacement_suggestions(courses: List[Course], all_courses_data: dict) -> List[str]:
    suggestions = []
    current_names = {c.name.lower() for c in courses}

    for course in courses:
        if (
            not course.code
            or course.hp >= 30.0
            or "examensarbete" in course.name.lower()
            or "projektkurs" in course.name.lower()
        ):
            continue

        # Skip if course is a prerequisite for others
        if any(course.name.lower() in c.prerequisites.lower() for c in courses):
            continue

        dependents = [c for c in courses if course.name.lower() in c.prerequisites.lower()]
        replacements = find_replacement_combinations(course, all_courses_data, current_names, dependents)

        if not replacements:
            continue

        block = []
        block.append(f"<<< REPLACEMENT FOR: {course.name} >>>")
        block.append(f"Original: [{course.code}] {course.category}: {course.name} ({course.hp} hp), Year {course.year}")
        block.append("Can be replaced with:")

        for alt in replacements[:5]:
            alt_courses, total_hp = alt

            # Deduplicate and sort
            unique = {}
            for a in alt_courses:
                code = a.get("kurskod")
                if code and a.get("is_active", True):
                    unique[code] = a

            sorted_alts = sorted(unique.values(), key=lambda x: x.get("giltig från", ""), reverse=True)

            for a in sorted_alts:
                block.append(f"* [{a['kurskod']}] {a.get('huvudområde(n)', 'Okänt')}: {a['name']} ({a['_parsed_hp']} hp)")

            # Add overhead/under note
            diff = total_hp - course.hp
            if abs(diff) > 0.01:
                note = f"  Note: {abs(diff):.1f} hp {'overhead' if diff > 0 else 'under (incomplete)'}"
                block.append(note)

        block.append("<<< END REPLACEMENT >>>")
        suggestions.append("\n".join(block))

    return suggestions

def extract_course_info(result: str, keyword: str) -> list[str]:
    lines = result.splitlines()
    matched_blocks = []
    current_block = []
    collecting = False
    keyword = keyword.lower()

    for line in lines:
        lower_line = line.lower().strip()

        # Start collecting block if keyword matches
        if (lower_line.startswith("<<< course:") or lower_line.startswith("<<< replacement for:")) and keyword in lower_line:
            collecting = True
            current_block = []
            continue

        # End of block
        if lower_line == "<<< end course >>>" or lower_line == "<<< end replacement >>>":
            if collecting:
                matched_blocks.append("\n".join(current_block).strip())
                current_block = []
                collecting = False
            continue

        if collecting:
            current_block.append(line)

    if not matched_blocks:
        return [f"No matches found for keyword: {keyword}"]

    return matched_blocks
      
def get_replacements_for_program(program_name: str, course_name) -> str:
    user_prompt = program_name
    # print(f"Mapping for : {user_prompt}")

    program = find_program_by_name(file_path, user_prompt)

    if program:
        # print(f"\nFound program: {program['name']} ({program.get('giltig från', 'okänd')})\n")

        innehall = program.get("innehåll", "")
        courses = extract_courses(innehall)

        # Build lookup dictionary for course name -> code
        code_lookup = build_course_code_lookup(course_path)
        
        # Assign course code to each course object
        for course in courses:
            name_lc = course.name.lower()
            entry = code_lookup.get(name_lc)

            if entry:
                course.code = entry.get("kurskod", "")
                course.metadata = entry
                course.prerequisites = entry.get("särskild behörighet", "")
                course.huvudomrade = entry.get("huvudområde(n)", "Okänt") or "Okänt"
            else:
                # Find best match by highest year
                best = None
                best_score = -1

                for full_name, info in code_lookup.items():
                    if name_lc in full_name:
                        score = parse_giltig_fran(info.get("giltig från", ""))
                        if score > best_score:
                            best = info
                            best_score = score

                if best:
                    course.code = best.get("kurskod", "")
                    course.metadata = best
                    course.prerequisites = best.get("särskild behörighet", "")
                    course.huvudomrade = best.get("huvudområde(n)", "Okänt") or "Okänt"

            # Log if still unmatched
            if not course.code:
                print(f"Still unmatched: {course.name}")

    else:
        print(f"Program '{user_prompt}' not found.")
        return
        
    # summarize_categories(courses)
    all_course_data = load_all_courses(course_path)
    all_course_data = {item.get("name", "").strip().lower(): item for item in all_course_data if item.get("name")}
    
    structure_blocks = format_courses_by_year(courses)
    replacement_blocks = collect_replacement_suggestions(courses, all_course_data)

    full_output_str = "\n".join(structure_blocks + replacement_blocks)
    full_output = extract_course_info(full_output_str, course_name)
    
    return "\n\n".join(full_output)

if __name__ == "__main__":
    "Sjuksköterskeprogrammet", "Lärande system"
    course_names = ["mekatronik"]
    target_program = ["Civilingenjörsprogrammet i robotik 2022"]
    result = get_replacements_for_program(target_program[0], course_names[0]) 
    print(result)
    