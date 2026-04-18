import sys
import task_manager as tm
import utils

MENU = """
=============================
       TASK MANAGER
=============================
  1. Add task
  2. List tasks
  3. Complete task
  4. Quit
============================="""

def prompt_add():
    raw = input("Task title: ")
    ok, result = utils.validate_title(raw)
    if not ok:
        print(utils.format_error(result))
        return
    tm.add_task(result)
    print(utils.format_success(f"'{result}' added."))

def prompt_list():
    print(utils.format_task_list(tm.tasks))

def prompt_complete():
    prompt_list()
    raw = input("Enter task ID to complete: ").strip()
    ok, result = utils.validate_task_id(raw, tm.tasks)
    if not ok:
        print(utils.format_error(result))
        return
    tm.complete_task(result)
    print(utils.format_success(f"Task {result} marked as complete."))

def main():
    while True:
        print(MENU)
        choice = input("Choose an option: ").strip()
        if choice == "1":
            prompt_add()
        elif choice == "2":
            prompt_list()
        elif choice == "3":
            prompt_complete()
        elif choice == "4":
            print("Goodbye!")
            sys.exit(0)
        else:
            print(utils.format_error("Invalid option. Choose 1–4."))

if __name__ == "__main__":
    main()
