def format_task(task):
    status = "x" if task["completed"] else " "
    return f"[{status}] {task['id']}. {task['title']}"


def format_task_list(tasks):
    if not tasks:
        return "No tasks found."
    return "\n".join(format_task(t) for t in tasks)


def validate_title(title):
    if not isinstance(title, str):
        return False, "Title must be a string."
    title = title.strip()
    if not title:
        return False, "Title cannot be empty."
    if len(title) > 200:
        return False, "Title cannot exceed 200 characters."
    return True, title


def validate_task_id(value):
    try:
        task_id = int(value)
    except (TypeError, ValueError):
        return False, "Task ID must be an integer."
    if task_id < 1:
        return False, "Task ID must be a positive integer."
    return True, task_id
