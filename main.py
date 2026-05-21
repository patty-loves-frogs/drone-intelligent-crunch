from core.runtime.runtime import DroneRuntime
from core.state.drone_state import state
import pprint


def main():

    runtime = DroneRuntime()

    print("=== DRONE SYSTEM STARTED ===")

    while True:

        user_prompt = input("\nMission drone (ou 'exit') : ")

        if user_prompt == "exit":
            print("Arrêt du système.")
            break

        result_state = runtime.run(user_prompt, state)

        # récupérer la dernière réponse du llm
        history = result_state["conversation"]["history"]
        last_response = next(
            (msg["message"] for msg in reversed(history) if msg["role"] == "llm"),
            "Pas de réponse"
        )

        print("\n--- RÉPONSE ---\n")
        print(last_response)

        print("\n--- STATE FINAL ---\n")
        pprint.pprint(result_state, sort_dicts=False)


if __name__ == "__main__":
    main()