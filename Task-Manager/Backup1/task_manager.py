tasks = []

def add_task(title):
    tasks.append({"id": len(tasks) + 1, "title": title, "completed": False})
    print(f"Added: {title}")

def list_tasks():
    if not tasks:
        print("No tasks.")
        return
    for task in tasks:
        status = "x" if task["completed"] else " "
        print(f"[{status}] {task['id']}. {task['title']}")

def complete_task(task_id):
    for task in tasks:
        if task["id"] == task_id:
            task["completed"] = True
            print(f"Completed: {task['title']}")
            return
    print(f"Task {task_id} not found.")

def main():
    print("Task Manager — commands: add <title>, list, done <id>, quit")
    while True:
        raw = input("> ").strip()
        if not raw:
            continue
        parts = raw.split(" ", 1)
        cmd = parts[0].lower()
        if cmd == "quit":
            break
        elif cmd == "add" and len(parts) == 2:
            add_task(parts[1])
        elif cmd == "list":
            list_tasks()
        elif cmd == "done" and len(parts) == 2 and parts[1].isdigit():
            complete_task(int(parts[1]))
        else:
            print("Usage: add <title> | list | done <id> | quit")

if __name__ == "__main__":
    main()
