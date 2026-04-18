tasks = []
next_id = 1


def add_task(title):
    global next_id
    task = {"id": next_id, "title": title, "completed": False}
    tasks.append(task)
    next_id += 1
    return task


def list_tasks():
    return list(tasks)


def complete_task(task_id):
    for task in tasks:
        if task["id"] == task_id:
            task["completed"] = True
            return task
    return None
