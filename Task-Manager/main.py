import task_manager
import utils


def menu():
    print("\n--- Task Manager ---")
    print("1. Add task")
    print("2. List tasks")
    print("3. Complete task")
    print("4. Quit")
    return input("Choose an option: ").strip()


def main():
    while True:
        choice = menu()

        if choice == "1":
            title = input("Task title: ")
            ok, result = utils.validate_title(title)
            if not ok:
                print(f"Error: {result}")
            else:
                task = task_manager.add_task(result)
                print(f"Added: {utils.format_task(task)}")

        elif choice == "2":
            tasks = task_manager.list_tasks()
            print("\n" + utils.format_task_list(tasks))

        elif choice == "3":
            value = input("Task ID to complete: ")
            ok, result = utils.validate_task_id(value)
            if not ok:
                print(f"Error: {result}")
            else:
                task = task_manager.complete_task(result)
                if task is None:
                    print(f"No task found with ID {result}.")
                else:
                    print(f"Completed: {utils.format_task(task)}")

        elif choice == "4":
            print("Bye!")
            break

        else:
            print("Invalid option, please choose 1-4.")


if __name__ == "__main__":
    main()
