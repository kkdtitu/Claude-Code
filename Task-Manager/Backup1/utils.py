MAX_TITLE_LENGTH = 100


# --- Output formatting ---

def format_task(task):
    status = "x" if task["completed"] else " "
    return f"[{status}] {task['id']}. {task['title']}"

def format_task_list(tasks):
    if not tasks:
        return "No tasks."
    return "\n".join(format_task(t) for t in tasks)

def format_success(message):
    return f"[OK] {message}"

def format_error(message):
    return f"[ERROR] {message}"


# --- Input validation ---

def validate_title(title):
    """Returns (True, cleaned_title) or (False, error_message)."""
    title = title.strip()
    if not title:
        return False, "Task title cannot be empty."
    if len(title) > MAX_TITLE_LENGTH:
        return False, f"Title too long (max {MAX_TITLE_LENGTH} chars)."
    return True, title

def validate_task_id(value, tasks):
    """Returns (True, int_id) or (False, error_message)."""
    if not value.isdigit():
        return False, f"'{value}' is not a valid task ID."
    task_id = int(value)
    if task_id < 1:
        return False, "Task ID must be a positive number."
    if not any(t["id"] == task_id for t in tasks):
        return False, f"Task {task_id} not found."
    return True, task_id
